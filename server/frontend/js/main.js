// @ts-check
// Global variable for precise/fast mode toggle
let preciseMode = false; // Default to Fast
let useLogScaleX = false; // For X-axis log scale
let lastPlottedVolumes = null; // To store volumes for replotting
let lastPlottedColors = null; // To store colors for replotting

/**
 * @param {string} msg
 * @returns {never}
 */
function assertNonNull(msg = "Non null assertion failed") {
  throw new Error(msg)
}

/**
 * @template T
 * @param {new (...args: any[]) => T} cls
 * @param {any} instance
 * @returns {T}
 */
function assertClass(cls, instance) {
  if (instance instanceof cls) {
    return instance
  }

  throw new Error(`"${instance}" is not of class "${cls}"`)
}

const canvas = assertClass(HTMLCanvasElement, document.querySelector("#file-display"))
const ctx = canvas.getContext("2d", { willReadFrequently: true }) ?? assertNonNull()

const offscreenCanvas = new OffscreenCanvas(canvas.width, canvas.height)
const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true }) ?? assertNonNull()

const vticks = document.querySelector("#cdf-vticks") ?? assertNonNull()
const lines = document.querySelector("#cdf-lines") ?? assertNonNull()
const points = document.querySelector("#cdf-points") ?? assertNonNull()

const loader = document.querySelector("#loader") ?? assertNonNull()
const cdfPlotSvg = assertClass(SVGSVGElement, document.querySelector("#cdf-plot-svg"))
const downloadCanvasBtn = assertClass(HTMLButtonElement, document.querySelector("#download-canvas-corner-btn"))
const downloadPlotCornerBtn = assertClass(HTMLButtonElement, document.querySelector("#download-plot-corner-btn"))

// New Radio Button References
const modeToggleInput = assertClass(HTMLInputElement, document.querySelector("#mode-toggle-input"));
const scaleToggleInput = assertClass(HTMLInputElement, document.querySelector("#scale-toggle-input"));
const xAxisScaleControlsPanel = assertClass(HTMLElement, document.querySelector("#x-axis-scale-controls-panel"));

const quantileElems = [
  {
    y: 500,
    prefix: "q10",
    line: document.querySelector("#q10-line") ?? assertNonNull(),
    text: document.querySelector("#q10-text") ?? assertNonNull(),
  },
  {
    y: 300,
    prefix: "q50",
    line: document.querySelector("#q50-line") ?? assertNonNull(),
    text: document.querySelector("#q50-text") ?? assertNonNull(),
  },
  {
    y: 100,
    prefix: "q90",
    line: document.querySelector("#q90-line") ?? assertNonNull(),
    text: document.querySelector("#q90-text") ?? assertNonNull(),
  },
]

// Draw the initial prompt
function drawPrompt() {
  const w = 1024
  const h = 768

  canvas.width = w
  canvas.height = h
  canvas.style.aspectRatio = `${w}/${h}`

  ctx.clearRect(0, 0, w, h)
  ctx.font = "lighter 48px sans-serif"
  ctx.textAlign = "center"
  ctx.fillStyle = "#ccc"
  ctx.fillText("Paste image or click to upload", canvas.width / 2, canvas.height / 2)
}

function clearCdf() {
  while (points.lastElementChild) {
    points.removeChild(points.lastElementChild)
  }

  lines.removeAttribute("d")

  for (const elem of quantileElems) {
    elem.text.textContent = ""
    elem.line.removeAttribute("x1")
    elem.line.removeAttribute("y1")
    elem.line.removeAttribute("x2")
    elem.line.removeAttribute("y2")

    // Remove associated background rects
    const parentG = elem.text.parentElement;
    if (parentG) {
      const rectsToRemove = parentG.querySelectorAll('.quantile-text-background');
      rectsToRemove.forEach(rect => parentG.removeChild(rect));
    }
  }
}

class Oklab {
  /**
   * @param {number} value
   */
  static gammaCorrect(value) {
    return value > 0.0031306684
      ? 1.055 * Math.pow(value, 1 / 2.4) - 0.055
      : 12.92 * value;
  }

  /**
   * @param {number} okl
   * @param {number} oka
   * @param {number} okb
   * @param {Uint8ClampedArray} out 
   * @returns {void}
   */
  static toRgb(okl, oka, okb, out) {
    const cbrtL = okl + 0.3963377774 * oka + 0.2158037573 * okb
    const cbrtM = okl - 0.1055613458 * oka - 0.0638541728 * okb
    const cbrtS = okl - 0.0894841775 * oka - 1.2914855480 * okb

    const l = cbrtL * cbrtL * cbrtL
    const m = cbrtM * cbrtM * cbrtM
    const s = cbrtS * cbrtS * cbrtS

    const r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    const b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    out[0] = Oklab.gammaCorrect(r) * 256
    out[1] = Oklab.gammaCorrect(g) * 256
    out[2] = Oklab.gammaCorrect(b) * 256
  }
}

/**
 * @param {number} count
 * @returns {Uint8ClampedArray}
 */
function randomColor(count) {
  const rand = new Uint32Array(count)
  crypto.getRandomValues(rand)

  const result = new Uint8ClampedArray(count * 3)
  for (let i = 0; i < count; ++i) {
    const h = rand[i] * Math.PI / 2147483648

    const sin = 0.25 * Math.sin(h)
    const cos = 0.25 * Math.cos(h)

    const view = result.subarray(i * 3, (i + 1) * 3)
    Oklab.toRgb(0.75, sin, cos, view)
  }

  return result
}

/**
 * @param {Float64Array} volumes
 * @param {Uint8ClampedArray} colors
 */
function plotCdf(volumes, colors) {
  if (!volumes || volumes.length === 0) {
    // Clear plot elements if no volumes
    while (points.lastElementChild) points.removeChild(points.lastElementChild);
    lines.removeAttribute("d");
    for (const elem of quantileElems) {
      elem.text.textContent = "";
      elem.line.removeAttribute("x1");
      elem.line.removeAttribute("y1");
      elem.line.removeAttribute("x2");
      elem.line.removeAttribute("y2");
      const parentG = elem.text.parentElement;
      if (parentG) {
        const rectsToRemove = parentG.querySelectorAll('.quantile-text-background');
        rectsToRemove.forEach(rect => parentG.removeChild(rect));
      }
    }
    const children = vticks.children;
    for (const child of children) child.textContent = "-";
    return;
  }

  // 1. Create an array of objects to hold original and display values
  const plotData = Array.from(volumes).map(v => ({
    originalValue: v,
    displayValue: useLogScaleX ? Math.log10(Math.max(v, 1)) : v
  }));

  // 2. Sort this array based on displayValue
  plotData.sort((a, b) => a.displayValue - b.displayValue);

  const minDisplayValue = plotData[0].displayValue;
  const maxDisplayValue = plotData[plotData.length - 1].displayValue;

  // 3. Update X-axis tick labels
  const tickElements = vticks.children;
  const numTicks = tickElements.length;
  for (let i = 0; i < numTicks; ++i) {
    const tickElement = tickElements[i];
    let tickValue;
    if (maxDisplayValue === minDisplayValue) {
        tickValue = useLogScaleX ? Math.pow(10, minDisplayValue) : minDisplayValue;
    } else {
        const proportion = i / (numTicks - 1);
        const currentDisplayValue = minDisplayValue + proportion * (maxDisplayValue - minDisplayValue);
        tickValue = useLogScaleX ? Math.pow(10, currentDisplayValue) : currentDisplayValue;
    }

    let notationOptions = {}; // Initialize as an empty object
    if (tickValue >= 1000 && tickValue < 1e7) { // 1,000 up to 10,000,000 (exclusive)
        notationOptions = {
            notation: 'compact',
            compactDisplay: 'short',
            minimumFractionDigits: 0,
            maximumFractionDigits: (tickValue % (tickValue >= 1e6 ? 1e6 : 1e3) === 0) ? 0 : 1
        };
    } else if (tickValue >= 1e7) { // 10,000,000 and above
        notationOptions = {
            notation: 'scientific',
            minimumFractionDigits: 0,
            maximumFractionDigits: 1 
        };
    } else { // Less than 1,000
        notationOptions = {
            notation: 'standard',
            minimumFractionDigits: 0,
            maximumFractionDigits: (tickValue < 100 && tickValue !== Math.floor(tickValue)) ? 2 : 0
        };
    }
    tickElement.textContent = tickValue.toLocaleString(undefined, notationOptions);
  }
  
  // Clear previous dynamic plot elements (points, main line, quantile backgrounds)
  while (points.lastElementChild) points.removeChild(points.lastElementChild);
  lines.removeAttribute("d");
  quantileElems.forEach(qe => {
    const parentG = qe.text.parentElement;
    if (parentG) {
        const rectsToRemove = parentG.querySelectorAll('.quantile-text-background');
        rectsToRemove.forEach(rect => parentG.removeChild(rect));
    }
  });

  const yDec = 500 / plotData.length;
  let y = 550;

  const xs = new Float64Array(plotData.length + 2);
  const ys = new Float64Array(plotData.length + 2);
  let distinctPointsIdx = 1; // Index for xs, ys (distinct points for drawing curve)

  let q_idx = 0;
  // Iterate through the sorted plotData for CDF points and quantile calculations
  for (let i = 0; i < plotData.length; ++i) {
    const dataPoint = plotData[i];
    y -= yDec;
    
    const x = (dataPoint.displayValue - minDisplayValue) * 600 / (maxDisplayValue - minDisplayValue) + 100;

    if (distinctPointsIdx > 0 && x === xs[distinctPointsIdx - 1]) {
      ys[distinctPointsIdx - 1] = y; // Update y for vertical segments
    } else {
      xs[distinctPointsIdx] = x;
      ys[distinctPointsIdx] = y;
      distinctPointsIdx += 1;
    }

    // Quantile logic (using originalValue for text)
    if (q_idx < quantileElems.length && y < quantileElems[q_idx].y) {
      const { line, text, prefix } = quantileElems[q_idx];
      line.setAttribute("x1", x.toString());
      line.setAttribute("x2", x.toString());
      line.setAttribute("y1", y.toString());
      line.setAttribute("y2", "550");

      text.textContent = `${prefix} = ${dataPoint.originalValue.toFixed(2)}`;
      text.setAttribute("y", y.toString());
      text.setAttribute("x", (x + 25).toString());
      
      const bgRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      bgRect.setAttribute("x", (x + 22).toString());
      bgRect.setAttribute("y", (y - 10).toString());
      bgRect.setAttribute("width", "100"); 
      bgRect.setAttribute("height", "20");
      bgRect.setAttribute("fill", "white");
      bgRect.setAttribute("fill-opacity", "0.7");
      bgRect.setAttribute("class", "quantile-text-background"); 
      const parent = text.parentElement;
      if (parent) parent.insertBefore(bgRect, text);
      
      ++q_idx;
    }
  }

  // Draw the CDF points
  for (let i = 1; i < distinctPointsIdx; ++i) {
    const elem = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    elem.setAttribute("cx", xs[i].toString());
    elem.setAttribute("cy", ys[i].toString());
    elem.setAttribute("r", "3");
    
    // Color mapping needs to be considered carefully if plotData was re-sorted.
    // If colors correspond to the original unsorted `volumes` array:
    // We need to find original index of plotData[i-1].originalValue in `volumes` to pick the right color.
    // For now, to keep it simpler, let's color based on the sorted order of plotData.
    // This means the (i-1)-th point in the sorted plot gets the (i-1)-th color.
    if (colors && (i-1) < colors.length / 3 ) {
      const colorIndex = (i-1) * 3; 
      const r_val = colors[colorIndex];
      const g_val = colors[colorIndex + 1];
      const b_val = colors[colorIndex + 2];
      elem.setAttribute("fill", `rgb(${r_val},${g_val},${b_val})`);
    } else {
      elem.setAttribute("fill", "black"); 
    }
    points.appendChild(elem);
  }

  if (distinctPointsIdx <= 2) {
    lines.removeAttribute("d"); // Clear path if not enough points
    return;
  }

  xs[0] = 2 * xs[1] - xs[2];
  ys[0] = 2 * ys[1] - ys[2];
  xs[distinctPointsIdx] = xs[distinctPointsIdx - 1] * 2 - xs[distinctPointsIdx - 2];
  ys[distinctPointsIdx] = ys[distinctPointsIdx - 1] * 2 - ys[distinctPointsIdx - 2];

  const sx = (xs[0] + 4 * xs[1] + xs[2]) / 6;
  const sy = (ys[0] + 4 * ys[1] + ys[2]) / 6;
  let d_path = `M ${sx} ${sy}\n`;

  for (let i = 2; i < distinctPointsIdx; ++i) {
    const c1x = (4 * xs[i - 1] + 2 * xs[i]) / 6;
    const c1y = (4 * ys[i - 1] + 2 * ys[i]) / 6;
    const c2x = (2 * xs[i - 1] + 4 * xs[i]) / 6;
    const c2y = (2 * ys[i - 1] + 4 * ys[i]) / 6;
    const nx = (xs[i - 1] + 4 * xs[i] + xs[i + 1]) / 6;
    const ny = (ys[i - 1] + 4 * ys[i] + ys[i + 1]) / 6;
    d_path += `C ${c1x} ${c1y} ${c2x} ${c2y} ${nx} ${ny}\n`;
  }
  lines.setAttribute("d", d_path);
}

/**
 * @param {Blob} blob
 * @param {number} sendTime
 * @param {number[]} [serverVolumes] - Volume data from server response
 */
function displayMask(blob, sendTime, serverVolumes) {
  const receiveTime = performance.now();
  console.log(`Time to receive response from server: ${(receiveTime - sendTime).toFixed(2)}ms`);
  
  const img = new Image()
  const src = URL.createObjectURL(blob)

  img.addEventListener("load", () => {
    const loadTime = performance.now();
    console.log(`Time to load mask image: ${(loadTime - receiveTime).toFixed(2)}ms`);
    
    const count = (img.height / img.width) >>> 0
    const colors = randomColor(count)

    // Set dimensions only once
    offscreenCanvas.width = canvas.width
    offscreenCanvas.height = canvas.height

    // Get source image data once
    const imgdata = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const datalen = imgdata.data.length

    const volumes = serverVolumes ? new Float64Array(serverVolumes) : new Float64Array(count)

    let sy = 0
    for (let i = 0; i < count; ++i) {
      offscreenCtx.drawImage(img, 0, sy, img.width, img.width, 0, 0, canvas.width, canvas.height)
      const maskData = offscreenCtx.getImageData(0, 0, canvas.width, canvas.height)
      const mask = maskData.data
      let pixelCount = 0
      const [r, g, b] = colors.subarray(i * 3, (i + 1) * 3)
      for (let idx = 0; idx < datalen; idx += 4) {
        const [maskR, maskG, maskB] = mask.subarray(idx, idx + 3)

        if (maskR >= 128 && maskB >= 128 && maskG >= 128) {
          pixelCount += 1
          const pixel = imgdata.data.subarray(idx, idx + 3)
          pixel[0] = (pixel[0] * 10 + r * 6) >>> 4
          pixel[1] = (pixel[1] * 10 + g * 6) >>> 4
          pixel[2] = (pixel[2] * 10 + b * 6) >>> 4
        }
      }

      if (!serverVolumes) {
        volumes[i] = 4/3 * Math.pow(pixelCount / Math.PI, 3/2)
      }
      sy += img.width
    }

    lastPlottedVolumes = Float64Array.from(volumes); // Store a copy of original volumes
    lastPlottedColors = Uint8ClampedArray.from(colors); // Store a copy of colors

    plotCdf(lastPlottedVolumes, lastPlottedColors)
    ctx.putImageData(imgdata, 0, 0)
    downloadCanvasBtn.disabled = false;
    downloadPlotCornerBtn.disabled = false;
    xAxisScaleControlsPanel.style.display = 'flex'; // Show X-axis scale controls
    scaleToggleInput.disabled = false;
    loader.classList.add("loader-hidden"); 
    
    const renderTime = performance.now();
    console.log(`Time to render results: ${(renderTime - loadTime).toFixed(2)}ms`);
    console.log(`Total time from send to complete render: ${(renderTime - sendTime).toFixed(2)}ms`);

    // Enable scale toggle and set its initial state (after processing)
    scaleToggleInput.disabled = false;
    xAxisScaleControlsPanel.style.display = 'flex'; // Show X-axis scale controls

    URL.revokeObjectURL(src)
  })

  img.src = src
}

// --- Download Functionality ---

/**
 * Downloads the content of the main canvas as a PNG image.
 */
function downloadCanvasImage() {
  if (canvas.width === 0 || canvas.height === 0 || downloadCanvasBtn.disabled) {
    console.warn("Canvas is empty or download is disabled.");
    return;
  }
  const dataURL = canvas.toDataURL("image/png");
  const link = document.createElement("a");
  link.href = dataURL;
  link.download = "segmented_image.png";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Downloads the CDF plot as an SVG file.
 */
function downloadPlotSVG() {
  if (downloadPlotCornerBtn.disabled) {
    console.warn("Plot download is disabled.");
    return;
  }
  const serializer = new XMLSerializer();
  const svgString = serializer.serializeToString(cdfPlotSvg);
  const dataURL = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svgString);
  const link = document.createElement("a");
  link.href = dataURL;
  link.download = "cdf_plot.svg";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Add event listeners for download buttons
downloadCanvasBtn.addEventListener("click", downloadCanvasImage);
downloadPlotCornerBtn.addEventListener("click", downloadPlotSVG);

/**
 * @param {File} file
 * Process the image file when the user uploaded or pasted
 */
function processFile(file) {
  clearCdf()
  loader.classList.remove("loader-hidden");

  // Convert to HTMLImageElement and send to canvas
  const reader = new FileReader()
  reader.addEventListener("load", () => {
    const src = reader.result?.toString()
    if (src == null) return

    const img = new Image()
    img.addEventListener("load", () => {
      const min = Math.min(img.width, img.height)
      const w = (img.width * 512 / min) >>> 0
      const h = (img.height * 512 / min) >>> 0

      // Resize and draw image to the canvas
      canvas.width = w
      canvas.height = h
      canvas.style.aspectRatio = `${w}/${h}`
      ctx.drawImage(img, 0, 0, w, h)

      // Send the image to the back-end
      canvas.toBlob(async blob => {
        if (blob == null) return

        const formdata = new FormData()
        formdata.append("file", blob, "input.png")
        
        const sendTime = performance.now();          
        console.log(`Sending image to server using RGBD model in ${preciseMode ? 'precise' : 'fast'} mode...`);
        const endpoint = `/predict?use_depth=${preciseMode ? 'precise' : 'fast'}`
        const resp = await fetch(endpoint, {
          method: "POST",
          body: formdata,
        })        
        const jsonResponse = await resp.json();
        
        if (!jsonResponse.success) {
          console.error("Error from server:", jsonResponse.error);
          alert(`Error processing image: ${jsonResponse.error}`);
          loader.classList.add("loader-hidden");
          return;
        }
        
        const imgBlob = await fetch(jsonResponse.image_data).then(r => r.blob());
        displayMask(imgBlob, sendTime, jsonResponse.volumes)
      }, "image/png")
    })
    img.src = src
  })
  reader.readAsDataURL(file)
}

// Handle file upload
const fileInput = assertClass(HTMLInputElement, document.querySelector("#file-input"))
fileInput.addEventListener("input", () => {
  const item = fileInput.files?.item(0)
  if (item == null) return
  processFile(item)
})

// Handle clipboard pasting
document.addEventListener("paste", e => {
  const items = e.clipboardData?.items

  if (items == null) return

  for (const item of items) {
    const file = item.getAsFile()
    if (file == null) continue

    processFile(file)
    break
  }
})

function reset() {
  clearCdf()
  drawPrompt()
  downloadCanvasBtn.disabled = true;
  downloadPlotCornerBtn.disabled = true;
  fileInput.value = "";
  lastPlottedVolumes = null;
  lastPlottedColors = null;
  xAxisScaleControlsPanel.style.display = 'none'; // Hide X-axis scale controls
  scaleToggleInput.disabled = true; // Disable scale toggle on reset
  // Ensure scale toggle is reset to Linear (unchecked)
  scaleToggleInput.checked = false;
  const scaleChangeEventInitial = new Event('change'); // Dispatch change to update text styles
  scaleToggleInput.dispatchEvent(scaleChangeEventInitial);
  
  plotCdf(new Float64Array(), new Uint8ClampedArray()); // Call with empty typed arrays
}

const form = assertClass(HTMLFormElement, document.querySelector("#main"))
form.addEventListener("submit", e => e.preventDefault())

// Add event listener for the new Clear button
const clearBtn = assertClass(HTMLButtonElement, document.querySelector('#clear-btn'));
clearBtn.addEventListener('click', reset);

// Event listener for the new mode toggle switch
modeToggleInput.addEventListener('change', (event) => {
    if (event.target instanceof HTMLInputElement) {
        preciseMode = event.target.checked; // true if checked (Precise), false if not (Fast)
        console.log(`Processing mode changed to: ${preciseMode ? 'Precise' : 'Fast'}`);
        // Update text styles based on preciseMode
        const fastText = assertClass(HTMLElement, document.querySelector('.mode-text-fast'));
        const preciseText = assertClass(HTMLElement, document.querySelector('.mode-text-precise'));
        if (fastText && preciseText) {
            if (preciseMode) {
                fastText.style.fontWeight = 'normal';
                preciseText.style.fontWeight = 'bold';
                // preciseText.style.color = '#4CAF50'; // Color is handled by CSS
            } else {
                fastText.style.fontWeight = 'bold';
                // fastText.style.color = '#333'; // Color is handled by CSS
                preciseText.style.fontWeight = 'normal';
                // preciseText.style.color = '#333'; // Color is handled by CSS
            }
        }
    }
});

// Event listener for the new scale toggle switch
scaleToggleInput.addEventListener('change', (event) => {
    if (event.target instanceof HTMLInputElement) {
        useLogScaleX = event.target.checked; // true if checked (Log), false if not (Linear)
        console.log(`X-axis scale changed to: ${useLogScaleX ? 'Log' : 'Linear'}`);
        
        // Update text styles based on useLogScaleX
        const linearText = assertClass(HTMLElement, document.querySelector('.scale-text-linear'));
        const logText = assertClass(HTMLElement, document.querySelector('.scale-text-log'));
        if (linearText && logText) {
            if (useLogScaleX) {
                linearText.style.fontWeight = 'normal';
                logText.style.fontWeight = 'bold';
            } else {
                linearText.style.fontWeight = 'bold';
                logText.style.fontWeight = 'normal';
            }
        }

        if (lastPlottedVolumes && lastPlottedColors) {
            plotCdf(lastPlottedVolumes, lastPlottedColors);
        }
    }
});

// Set initial states from global vars
modeToggleInput.checked = preciseMode;
// Manually trigger the change event to apply initial styles for the mode toggle text
const modeChangeEvent = new Event('change');
modeToggleInput.dispatchEvent(modeChangeEvent);

scaleToggleInput.checked = useLogScaleX;
scaleToggleInput.disabled = true; // Initially disabled until an image is processed
// Manually trigger the change event to apply initial styles for the scale toggle text
const scaleChangeEventInitial = new Event('change');
scaleToggleInput.dispatchEvent(scaleChangeEventInitial);

reset()
