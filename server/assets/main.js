// @ts-check

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

const offscreenCanvas = new OffscreenCanvas(canvas.width, canvas.height)
const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true }) ?? assertNonNull()

const vticks = document.querySelector("#cdf-vticks") ?? assertNonNull()
const lines = document.querySelector("#cdf-lines") ?? assertNonNull()
const points = document.querySelector("#cdf-points") ?? assertNonNull()

/**
 * @param {Float64Array} areas
 */
function plotCdf(areas) {
  areas.sort()

  const min = areas[0]
  const max = areas[areas.length - 1]

  // Draw number on horizontal ticks
  const children = vticks.children
  let tick = min
  const inc = (max - min) / (children.length - 1)
  for (const child of children) {
    child.textContent = tick.toFixed(0)
    tick += inc
  }

  while (points.childElementCount) {
    points.removeChild(points.lastElementChild ?? assertNonNull())
  }

  const yDec = 500 / areas.length
  let y = 550

  // Compute the points of the CDF plot
  const xs = new Float64Array(areas.length + 2)
  const ys = new Float64Array(areas.length + 2)
  let idx = 1
  for (const area of areas) {
    y -= yDec
    const x = (area - min) * 600 / (max - min) + 100
    if (idx > 0 && x == xs[idx - 1]) {
      ys[idx - 1] = y
    } else {
      xs[idx] = x
      ys[idx] = y
      idx += 1
    }
  }

  // Draw the points
  for (let i = 0; i < idx; ++i) {
    // TODO: Fill each point with their respective color
    const elem = document.createElementNS("http://www.w3.org/2000/svg", "circle")
    elem.setAttribute("cx", xs[i + 1].toString())
    elem.setAttribute("cy", ys[i + 1].toString())
    elem.setAttribute("r", "5")
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
 */
function displayMask(blob) {
  const img = new Image()
  const src = URL.createObjectURL(blob)

  img.addEventListener("load", () => {
    const count = (img.height / img.width) >>> 0
    const colors = randomColor(count)

    const w = canvas.width
    const h = canvas.height * count
    offscreenCanvas.width = w
    offscreenCanvas.height = h

    offscreenCtx.drawImage(img, 0, 0, w, h)
    const maskData = offscreenCtx.getImageData(0, 0, w, h)

    const imgdata = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const datalen = imgdata.data.length

    const areas = new Float64Array(count)

    let mask = maskData.data
    for (let i = 0; i < count; ++i) {
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

      mask = mask.subarray(datalen)
      areas[i] = 4/3 * Math.pow(pixelCount / Math.PI, 3/2)
    }

    plotCdf(areas)

    ctx.putImageData(imgdata, 0, 0)

    URL.revokeObjectURL(src)
  })

  img.src = src
}

/**
 * @param {File} file
 * Process the image file when the user uploaded or pasted
 */
function processFile(file) {
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

        const resp = await fetch("/predict", {
          method: "POST",
          body: formdata,
        })

        displayMask(await resp.blob())
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

const form = assertClass(HTMLFormElement, document.querySelector("#main"))
form.addEventListener("submit", e => e.preventDefault())
form.addEventListener("reset", () => {
  while (points.lastElementChild) {
    points.removeChild(points.lastElementChild)
  }
  lines.removeAttribute("d")
  drawPrompt()
})

drawPrompt()
