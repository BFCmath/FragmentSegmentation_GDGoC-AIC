// @ts-check
// Global variable for precise/fast mode toggle
let preciseMode = false;

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

const downloadBtn = document.querySelector("#download-btn") ?? assertNonNull()

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
 */
function plotCdf(volumes) {
  volumes.sort()

  const min = volumes[0]
  const max = volumes[volumes.length - 1]

  // Draw number on horizontal ticks
  const children = vticks.children
  let tick = min
  const inc = (max - min) / (children.length - 1)
  for (const child of children) {
    child.textContent = tick.toFixed(0)
    tick += inc
  }
  
  // Clear any existing points first
  while (points.childElementCount) {
    throw new Error("CDF plot is not empty") 
  }

  const yDec = 500 / volumes.length
  let y = 550

  // Compute the points of the CDF plot
  const xs = new Float64Array(volumes.length + 2)
  const ys = new Float64Array(volumes.length + 2)
  let idx = 1

  let q_idx = 0
  for (const area of volumes) {
    y -= yDec
    const x = (area - min) * 600 / (max - min) + 100
    if (idx > 0 && x == xs[idx - 1]) {
      ys[idx - 1] = y
    } else {
      xs[idx] = x
      ys[idx] = y
      idx += 1
    }

    if (q_idx < quantileElems.length && y < quantileElems[q_idx].y) {
      const { line, text, prefix } = quantileElems[q_idx]
      line.setAttribute("x1", x.toString())
      line.setAttribute("x2", x.toString())
      line.setAttribute("y1", y.toString())
      line.setAttribute("y2", "550")

      text.textContent = `${prefix} = ${area.toFixed(2)}`
      text.setAttribute("x", (x + 10).toString())
      text.setAttribute("transform", `rotate(-90 ${x + 10} 540)`)
      ++q_idx
    }
  }

  // Draw the points
  for (let i = 1; i < idx; ++i) {
    // TODO: Fill each point with their respective color
    const elem = document.createElementNS("http://www.w3.org/2000/svg", "circle")
    elem.setAttribute("cx", xs[i].toString())
    elem.setAttribute("cy", ys[i].toString())
    elem.setAttribute("r", "3")
    points.appendChild(elem)
  }

  if (idx <= 2) return

  // Define extra points at the start and end for interpolation
  xs[0] = 2 * xs[1] - xs[2]
  ys[0] = 2 * ys[1] - ys[2]
  xs[idx] = xs[idx - 1] * 2 - xs[idx - 2]
  ys[idx] = ys[idx - 1] * 2 - ys[idx - 2]


  // Draw the approximation line
  const sx = (xs[0] + 4 * xs[1] + xs[2]) / 6
  const sy = (ys[0] + 4 * ys[1] + ys[2]) / 6
  let d = `M ${sx} ${sy}\n`

  for (let i = 2; i < idx; ++i) {
    const c1x = (4 * xs[i - 1] + 2 * xs[i]) / 6
    const c1y = (4 * ys[i - 1] + 2 * ys[i]) / 6

    const c2x = (2 * xs[i - 1] + 4 * xs[i]) / 6
    const c2y = (2 * ys[i - 1] + 4 * ys[i]) / 6

    const nx = (xs[i - 1] + 4 * xs[i] + xs[i + 1]) / 6
    const ny = (ys[i - 1] + 4 * ys[i] + ys[i + 1]) / 6
    d += `C ${c1x} ${c1y} ${c2x} ${c2y} ${nx} ${ny}\n`
  }

  lines.setAttribute("d", d)
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

    // Use server-provided volumes if available, otherwise create empty array
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

      // Only calculate volumes if not provided by server
      if (!serverVolumes) {
        volumes[i] = 4/3 * Math.pow(pixelCount / Math.PI, 3/2)
      }
      sy += img.width
    }

    plotCdf(volumes)
    ctx.putImageData(imgdata, 0, 0)
    downloadBtn.removeAttribute("data-disabled")
    
    const renderTime = performance.now();
    console.log(`Time to render results: ${(renderTime - loadTime).toFixed(2)}ms`);
    console.log(`Total time from send to complete render: ${(renderTime - sendTime).toFixed(2)}ms`);

    URL.revokeObjectURL(src)
  })

  img.src = src
}

/**
 * @param {File} file
 * Process the image file when the user uploaded or pasted
 */
function processFile(file) {
  clearCdf()

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
  downloadBtn.setAttribute("data-disabled", "disabled")
}

const form = assertClass(HTMLFormElement, document.querySelector("#main"))
form.addEventListener("submit", e => e.preventDefault())
form.addEventListener("reset", reset)

// Add event listener for the mode toggle button
const modeToggleBtn = document.querySelector('#mode-toggle-btn');
if (modeToggleBtn) {
  modeToggleBtn.addEventListener('click', () => {
    preciseMode = !preciseMode;
    modeToggleBtn.textContent = preciseMode ? 'Switch to Fast Mode' : 'Switch to Precise Mode';
  });
}

reset()
