import * as THREE from "three";

let cached: THREE.CanvasTexture | null = null;

/**
 * The crew patch: the Atheera logo on a white disc with a gold rim.
 * It is the project's own logo, not an agency insignia. The disc shows at once and the logo is drawn in as soon as the image has loaded.
 */
export function missionPatch(): THREE.CanvasTexture {
  if (cached) return cached;
  const S = 256, mid = S / 2;
  const c = document.createElement("canvas");
  c.width = c.height = S;
  const g = c.getContext("2d")!;

  g.fillStyle = "#c9a24a";
  g.beginPath(); g.arc(mid, mid, mid - 2, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#ffffff";
  g.beginPath(); g.arc(mid, mid, mid - 16, 0, Math.PI * 2); g.fill();

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;

  const img = new Image();
  img.onload = () => {
    const w = S - 76, h = (w * img.naturalHeight) / img.naturalWidth;
    g.drawImage(img, (S - w) / 2, (S - h) / 2, w, h);
    tex.needsUpdate = true;
  };
  img.src = "/logo.png";

  cached = tex;
  return tex;
}
