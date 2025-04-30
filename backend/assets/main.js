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

  throw new Error(`'${instance}' is not of class '${cls}'`)
}


const canvas = assertClass(HTMLCanvasElement, document.querySelector("#file-display"))
const ctx = canvas.getContext('2d') ?? assertNonNull()

// Draw the initial prompt
ctx.font = 'lighter 48px sans-serif'
ctx.textAlign = 'center'
ctx.fillStyle = '#ccc'
ctx.fillText('Paste image or click to upload', canvas.width / 2, canvas.height / 2)


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
   */
  static toRgb(okl, oka, okb) {
    const l = Math.pow(okl + 0.3963377774 * oka + 0.2158037573 * okb, 3);
    const m = Math.pow(okl - 0.1055613458 * oka - 0.0638541728 * okb, 3);
    const s = Math.pow(okl - 0.0894841775 * oka - 1.2914855480 * okb, 3);

    const r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    const b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    return [
      Oklab.gammaCorrect(r),
      Oklab.gammaCorrect(g),
      Oklab.gammaCorrect(b),
    ];
  }
}

/**
 * @param {number} count
 * @returns {number[][]}
 */
function randomColor(count) {
  const h = new Uint32Array(count)
  crypto.getRandomValues(h)

  /** @type{number[][]} */
  const result = new Array(count)
  for (let i = 0; i < count; ++i) {
    const r = h[i] * Math.PI / 2147483648

    const a = 0.25 * Math.sin(r)
    const b = 0.25 * Math.cos(r)
    result[i] = Oklab.toRgb(0.75, a, b)
  }

  return result
}


/**
 * @param {Blob} blob
 */
function displayMask(blob) {
  const img = new Image()
  const src = URL.createObjectURL(blob)
  
  const imgdata = ctx.getImageData(0, 0, canvas.width, canvas.height)

  img.addEventListener('load', () => {
    const count = (img.height / img.width) >>> 0
    const colors = randomColor(count)

    let sy = 0;
    for (let i = 0; i < count; ++i) {
      ctx.drawImage(img, 0, sy, img.width, img.width, 0, 0, canvas.width, canvas.height)
      const mask_data = ctx.getImageData(0, 0, canvas.width, canvas.height)

      for (let y = 0; y < canvas.height; ++y) {
        for (let x = 0; x < canvas.width; ++x) {
          const idx = 4 * (canvas.width * y + x)

          const mask_r = mask_data.data[idx + 0]
          const mask_b = mask_data.data[idx + 1]
          const mask_g = mask_data.data[idx + 2]

          if (mask_r > 0 || mask_b > 0 || mask_g > 0) {
            const [r, g, b] = colors[i]
            imgdata.data[idx + 0] = imgdata.data[idx + 0] * 0.6 + r * 102
            imgdata.data[idx + 1] = imgdata.data[idx + 1] * 0.6 + g * 102
            imgdata.data[idx + 2] = imgdata.data[idx + 2] * 0.6 + b * 102
          }
        }
      }

      sy += img.width
    }

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
  reader.addEventListener('load', () => {
    const src = reader.result?.toString()
    if (src == null) return

    const img = new Image()
    img.addEventListener('load', () => {
      const min = Math.min(img.width, img.height)
      const w = (img.width * 512 / min) >>> 0
      const h = (img.height * 512 / min) >>> 0

      // Resize and draw image to the canvas
      canvas.width = w
      canvas.height = h
      canvas.style.aspectRatio = w + '/' + h
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
const file_input = assertClass(HTMLInputElement, document.querySelector("#file-input"))
file_input.addEventListener('input', () => {
  const item = file_input.files?.item(0)
  if (item == null) return
  processFile(item)
})

// Handle clipboard pasting
document.addEventListener('paste', e => {
  const items = e.clipboardData?.items

  if (items == null) return

  for (const item of items) {
    const file = item.getAsFile()
    if (file == null) continue

    processFile(file)
    break
  }
})
