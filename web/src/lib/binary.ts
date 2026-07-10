/** Fetch a float32 array served by server/arrays.py (X-Shape header + raw bytes). */
export interface Float32Matrix {
  data: Float32Array
  shape: number[]
}

export async function fetchFloat32(url: string): Promise<Float32Matrix> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: ${res.status}`)
  const shapeHeader = res.headers.get('X-Shape')
  if (!shapeHeader) throw new Error(`${url}: missing X-Shape header`)
  const shape = shapeHeader.split(',').map(Number)
  const data = new Float32Array(await res.arrayBuffer())
  return { data, shape }
}
