"use client";

import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { forwardRef, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useInView } from "./in-view";
import { missionPatch } from "./patch";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { liveWaveSample, type CrewHistory } from "@/lib/telemetry";
import type { Frame } from "@/lib/types";

/* ------------------------------------------------------------------ station geometry (schematic, not to scale) */
type Axis = "x" | "z";
interface Mod3D { pos: [number, number, number]; axis: Axis; r: number; len: number; label?: string }

export const MODULES_3D: Record<string, Mod3D> = {
  CQ1: { pos: [-7.2, 0, -3.3], axis: "z", r: 1.1, len: 3.0, label: "Crew quarters 1" },
  CQ2: { pos: [-7.2, 0, 3.3], axis: "z", r: 1.1, len: 3.0, label: "Crew quarters 2" },
  CQ3: { pos: [-10.0, 0, -3.3], axis: "z", r: 1.1, len: 3.0, label: "Crew quarters 3" },
  CQ4: { pos: [-10.0, 0, 3.3], axis: "z", r: 1.1, len: 3.0, label: "Crew quarters 4" },
  NODE2: { pos: [-8.6, 0, 0], axis: "x", r: 1.5, len: 5.6 },
  LAB: { pos: [0.6, 0, 0], axis: "x", r: 1.6, len: 8.6, label: "US Lab" },
  NODE1: { pos: [8.5, 0, 0], axis: "x", r: 1.6, len: 3.6, label: "Node 1 galley" },
};
const CONNECTORS: { from: [number, number, number]; to: [number, number, number]; r: number }[] = [
  { from: [-5.8, 0, 0], to: [-3.7, 0, 0], r: 0.75 },
  { from: [4.9, 0, 0], to: [6.7, 0, 0], r: 0.75 },
];

export interface Palette {
  hull: string; edge: string; panel: string; accent: string;
  nominal: string; advisory: string; caution: string; warning: string; emergency: string; bag: string; suit: string; shirt: string;
}

export function readPalette(): Palette {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string, fb: string) => cs.getPropertyValue(n).trim() || fb;
  return {
    hull: "#dfe5ee", edge: "#8d9ab0", panel: "#f4f6fa", accent: v("--brand", "#0264cc"),
    nominal: v("--nominal", "#2f8a5b"), advisory: v("--advisory", "#6c7d91"), caution: v("--caution", "#d39a28"),
    warning: v("--warning", "#e0663a"), emergency: v("--emergency", "#d6403b"),
    bag: v("--sleep-bag", "#5b74a0"), suit: "#eef1f6", shirt: "#0a4aa6",
  };
}

const SEV_RANK: Record<string, number> = { NOMINAL: 0, ADVISORY: 1, CAUTION: 2, WARNING: 3, EMERGENCY: 4 };
const statusColor = (p: Palette, s: string) =>
  ({ NOMINAL: p.nominal, ADVISORY: p.advisory, CAUTION: p.caution, WARNING: p.warning, EMERGENCY: p.emergency } as Record<string, string>)[s] ?? p.nominal;

const axisRotation = (a: Axis): [number, number, number] => (a === "x" ? [0, 0, Math.PI / 2] : [Math.PI / 2, 0, 0]);
/** Interior props are modelled along +x; z-axis modules turn them a quarter. */
const propRotation = (a: Axis): [number, number, number] => (a === "x" ? [0, 0, 0] : [0, Math.PI / 2, 0]);
const floorY = (m: Mod3D) => m.pos[1] - m.r * 0.78;

/* ------------------------------------------------------------------ space: stars and Earth */
function Stars() {
  const geo = useMemo(() => {
    const n = 2200, pts = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, r = 260;
      const s = Math.sqrt(1 - u * u);
      pts.set([r * s * Math.cos(th), r * u, r * s * Math.sin(th)], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pts, 3));
    return g;
  }, []);
  return (
    <points geometry={geo}>
      <pointsMaterial color="#dfe8ff" size={1.4} sizeAttenuation={false} transparent opacity={0.85} depthWrite={false} />
    </points>
  );
}

/* procedural value noise, enough for continents and clouds */
function makeNoise(seed: number) {
  const hash = (x: number, y: number) => {
    const s = Math.sin(x * 127.1 + y * 311.7 + seed * 74.7) * 43758.5453;
    return s - Math.floor(s);
  };
  const smooth = (t: number) => t * t * (3 - 2 * t);
  const noise = (x: number, y: number, wrap: number) => {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const w = (a: number) => ((a % wrap) + wrap) % wrap;
    const a = hash(w(xi), yi), b = hash(w(xi + 1), yi), c = hash(w(xi), yi + 1), d = hash(w(xi + 1), yi + 1);
    const u = smooth(xf), v = smooth(yf);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
  return (x: number, y: number) => {
    let amp = 0.5, f = 1, sum = 0;
    for (let o = 0; o < 5; o++) { sum += amp * noise(x * f, y * f, 8 * f); amp *= 0.5; f *= 2; }
    return sum;
  };
}

function earthTextures() {
  const W = 512, H = 256;
  const land = makeNoise(3), cloud = makeNoise(11);
  const c1 = document.createElement("canvas"); c1.width = W; c1.height = H;
  const c2 = document.createElement("canvas"); c2.width = W; c2.height = H;
  const g1 = c1.getContext("2d")!, g2 = c2.getContext("2d")!;
  const im1 = g1.createImageData(W, H), im2 = g2.createImageData(W, H);
  for (let y = 0; y < H; y++) {
    const lat = Math.abs(y / H - 0.5) * 2;
    for (let x = 0; x < W; x++) {
      const nx = (x / W) * 8, ny = (y / H) * 4;
      const h = land(nx, ny) - lat * 0.08;
      const i = (y * W + x) * 4;
      let r: number, g: number, b: number;
      if (h > 0.58) {
        const k = Math.min(1, (h - 0.58) * 5);
        r = 70 + 90 * k; g = 112 + 40 * k; b = 62 + 30 * k;
        if (lat < 0.35 && h > 0.6) { r = 176; g = 152; b = 104; }
      } else {
        const k = Math.max(0, Math.min(1, (0.58 - h) * 4));
        r = 28 - 16 * k; g = 92 - 40 * k; b = 168 - 40 * k;
      }
      im1.data.set([r, g, b, 255], i);
      const cl = cloud(nx * 1.3 + 3, ny * 1.3);
      const a = Math.max(0, Math.min(1, (cl - 0.5) * 3.2));
      im2.data.set([255, 255, 255, Math.round(a * 220)], i);
    }
  }
  g1.putImageData(im1, 0, 0); g2.putImageData(im2, 0, 0);
  const t1 = new THREE.CanvasTexture(c1), t2 = new THREE.CanvasTexture(c2);
  t1.colorSpace = THREE.SRGBColorSpace;
  return { surface: t1, clouds: t2 };
}

function Earth({ still }: { still: boolean }) {
  const tex = useMemo(earthTextures, []);
  const ground = useRef<THREE.Mesh>(null), sky = useRef<THREE.Mesh>(null);
  useFrame((_, dt) => {
    if (still) return;
    if (ground.current) ground.current.rotation.y += dt * 0.012;
    if (sky.current) sky.current.rotation.y += dt * 0.016;
  });
  const R = 90, C: [number, number, number] = [0, -122, -45];
  return (
    <group position={C} rotation={[-0.5, 0, 0.12]}>
      <mesh ref={ground}>
        <sphereGeometry args={[R, 96, 64]} />
        <meshStandardMaterial map={tex.surface} roughness={0.9} metalness={0} />
      </mesh>
      <mesh ref={sky}>
        <sphereGeometry args={[R * 1.006, 96, 64]} />
        <meshStandardMaterial map={tex.clouds} transparent depthWrite={false} roughness={1} />
      </mesh>
      <mesh>
        <sphereGeometry args={[R * 1.035, 64, 48]} />
        <meshBasicMaterial color="#6fa4ff" transparent opacity={0.22} side={THREE.BackSide} blending={THREE.AdditiveBlending} depthWrite={false} />
      </mesh>
    </group>
  );
}

/* ------------------------------------------------------------------ exterior: truss, solar wings, radiators */
function SolarWing({ x, still }: { x: number; still: boolean }) {
  const ref = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    if (ref.current && !still) ref.current.rotation.z = Math.sin(clock.elapsedTime * 0.05 + x) * 0.35;
  });
  const cells = useMemo(() => Array.from({ length: 8 }, (_, k) => -10.5 - k * 0.95), []);
  return (
    <group position={[x, 0.6, 0]}>
      <group ref={ref}>
        <mesh position={[0, 0, -14]}>
          <boxGeometry args={[2.1, 0.04, 7.8]} />
          <meshStandardMaterial color="#1d2f63" metalness={0.5} roughness={0.35} emissive="#0b1838" emissiveIntensity={0.4} />
        </mesh>
        {cells.map((z) => (
          <mesh key={z} position={[0, 0.03, z]}>
            <boxGeometry args={[2.14, 0.02, 0.05]} />
            <meshStandardMaterial color="#c8a453" metalness={0.8} roughness={0.3} />
          </mesh>
        ))}
        <mesh position={[0, 0.03, -14]}>
          <boxGeometry args={[0.06, 0.03, 7.8]} />
          <meshStandardMaterial color="#c8a453" metalness={0.8} roughness={0.3} />
        </mesh>
      </group>
    </group>
  );
}

function Exterior({ still }: { still: boolean }) {
  // the truss runs behind the pressurised modules so it never hides the crew
  return (
    <group>
      <mesh position={[0.6, 0.6, -9]}>
        <boxGeometry args={[26, 0.6, 0.6]} />
        <meshStandardMaterial color="#c9d0db" metalness={0.6} roughness={0.45} wireframe />
      </mesh>
      <mesh position={[0.6, 0.6, -9]}>
        <boxGeometry args={[26, 0.2, 0.2]} />
        <meshStandardMaterial color="#9aa6b8" metalness={0.7} roughness={0.4} />
      </mesh>
      <mesh position={[0.6, 0.3, -5.4]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.14, 0.14, 7.2, 12]} />
        <meshStandardMaterial color="#9aa6b8" metalness={0.7} roughness={0.4} />
      </mesh>
      {[-10.5, -7.6, 8.8, 11.7].map((x) => <SolarWing key={x} x={x} still={still} />)}
      {[-2.4, 3.6].map((x) => (
        <mesh key={x} position={[x, 0.6, -11.2]}>
          <boxGeometry args={[1.6, 0.04, 3.4]} />
          <meshStandardMaterial color="#f2f4f8" roughness={0.6} />
        </mesh>
      ))}
    </group>
  );
}

/* ------------------------------------------------------------------ hull (cutaway: the near wall is never drawn) */
function Hull({ m, pal }: { m: Mod3D; pal: Palette }) {
  const rings = useMemo(() => {
    const out: number[] = [];
    const n = Math.max(2, Math.round(m.len / 1.1));
    for (let k = 1; k < n; k++) out.push(-m.len / 2 + (k * m.len) / n);
    return out;
  }, [m.len]);
  return (
    <group position={m.pos} rotation={axisRotation(m.axis)}>
      {/* inside walls: back faces only, so whichever side faces the camera disappears */}
      <mesh>
        <cylinderGeometry args={[m.r, m.r, m.len, 56, 1, true]} />
        <meshStandardMaterial color={pal.hull} side={THREE.BackSide} roughness={0.85} />
      </mesh>
      {/* faint outer skin for the silhouette */}
      <mesh>
        <cylinderGeometry args={[m.r + 0.02, m.r + 0.02, m.len, 56, 1, true]} />
        <meshStandardMaterial color="#ffffff" transparent opacity={0.1} depthWrite={false} roughness={0.4} />
      </mesh>
      {rings.map((y) => (
        <mesh key={y} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[m.r + 0.01, 0.02, 6, 72]} />
          <meshStandardMaterial color={pal.edge} transparent opacity={0.6} />
        </mesh>
      ))}
      {[-m.len / 2, m.len / 2].map((y) => (
        <mesh key={y} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[m.r + 0.02, 0.06, 10, 72]} />
          <meshStandardMaterial color={pal.edge} metalness={0.5} roughness={0.4} />
        </mesh>
      ))}
    </group>
  );
}

function Connector({ c, pal }: { c: (typeof CONNECTORS)[number]; pal: Palette }) {
  const len = Math.abs(c.to[0] - c.from[0]);
  return (
    <group position={[(c.from[0] + c.to[0]) / 2, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
      <mesh>
        <cylinderGeometry args={[c.r, c.r, len, 32, 1, true]} />
        <meshStandardMaterial color={pal.hull} side={THREE.BackSide} roughness={0.85} />
      </mesh>
      <mesh>
        <cylinderGeometry args={[c.r + 0.02, c.r + 0.02, len, 32, 1, true]} />
        <meshStandardMaterial color="#ffffff" transparent opacity={0.1} depthWrite={false} />
      </mesh>
    </group>
  );
}

/* ------------------------------------------------------------------ interior: racks, lights and module props */
function Racks({ m, pal }: { m: Mod3D; pal: Palette }) {
  const n = Math.max(2, Math.floor(m.len / 1.05));
  const w = (m.len * 0.9) / n;
  const lights = useMemo(() => Array.from({ length: n }, (_, k) => ["#4ade80", "#60a5fa", "#fbbf24", "#f87171"][(k * 7 + m.len * 3) % 4 | 0]), [n, m.len]);
  return (
    <group position={m.pos} rotation={propRotation(m.axis)}>
      {Array.from({ length: n }).map((_, k) => {
        const x = -m.len * 0.45 + w * (k + 0.5);
        return (
          <group key={k}>
            {/* floor rack */}
            <mesh position={[x, -m.r * 0.78, 0]}>
              <boxGeometry args={[w * 0.94, 0.26, m.r * 1.05]} />
              <meshStandardMaterial color={k % 2 ? "#e4e8ef" : "#d7dde7"} roughness={0.7} />
            </mesh>
            {/* back-wall rack with a status light */}
            <mesh position={[x, 0, -m.r * 0.8]}>
              <boxGeometry args={[w * 0.94, m.r * 1.05, 0.22]} />
              <meshStandardMaterial color={k % 2 ? "#dde2ea" : "#e8ecf2"} roughness={0.7} />
            </mesh>
            <mesh position={[x - w * 0.3, m.r * 0.3, -m.r * 0.8 + 0.12]}>
              <boxGeometry args={[0.07, 0.07, 0.02]} />
              <meshBasicMaterial color={lights[k]} />
            </mesh>
            <mesh position={[x + w * 0.1, -m.r * 0.05, -m.r * 0.8 + 0.12]}>
              <boxGeometry args={[w * 0.5, 0.26, 0.02]} />
              <meshStandardMaterial color="#26324a" roughness={0.4} emissive={pal.accent} emissiveIntensity={0.12} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

function Treadmill({ m }: { m: Mod3D }) {
  const belt = useRef<THREE.Mesh>(null);
  return (
    <group position={[m.pos[0], floorY(m) + 0.2, m.pos[2]]}>
      <mesh>
        <boxGeometry args={[1.5, 0.14, 0.7]} />
        <meshStandardMaterial color="#3b4458" roughness={0.6} />
      </mesh>
      <mesh ref={belt} position={[0, 0.08, 0]}>
        <boxGeometry args={[1.35, 0.02, 0.55]} />
        <meshStandardMaterial color="#141a26" roughness={0.9} />
      </mesh>
      <mesh position={[0.68, 0.55, 0]}>
        <boxGeometry args={[0.06, 0.9, 0.6]} />
        <meshStandardMaterial color="#9aa6b8" metalness={0.5} roughness={0.4} />
      </mesh>
    </group>
  );
}

function Workbench({ m }: { m: Mod3D }) {
  return (
    <group position={[m.pos[0] + 1.6, m.pos[1], m.pos[2] - m.r * 0.55]}>
      <mesh position={[0, -0.15, 0]}>
        <boxGeometry args={[1.4, 0.06, 0.5]} />
        <meshStandardMaterial color="#c6ceda" roughness={0.6} />
      </mesh>
      <mesh position={[0, 0.05, -0.12]} rotation={[-0.3, 0, 0]}>
        <boxGeometry args={[0.55, 0.36, 0.03]} />
        <meshStandardMaterial color="#1b2333" emissive="#3a6bdc" emissiveIntensity={0.35} />
      </mesh>
    </group>
  );
}

/** Module light: warm white normally; pulses in the colour of the worst alert inside. */
function ModuleLight({ id, m, pal, occupants, framesRef, still }: {
  id: string; m: Mod3D; pal: Palette; occupants: () => string[]; framesRef: React.RefObject<Record<string, Frame>>; still: boolean;
}) {
  const light = useRef<THREE.PointLight>(null);
  const strip = useRef<THREE.MeshBasicMaterial>(null);
  const col = useMemo(() => new THREE.Color(), []);
  const warm = useMemo(() => new THREE.Color("#fff3df"), []);
  useFrame(({ clock }) => {
    let worst = "NOMINAL";
    for (const c of occupants()) {
      const s = framesRef.current[c]?.status ?? "NOMINAL";
      if (SEV_RANK[s] > SEV_RANK[worst]) worst = s;
    }
    const alarm = SEV_RANK[worst] >= 3;
    const pulse = still ? 1 : 0.5 + 0.5 * Math.sin(clock.elapsedTime * 6);
    col.copy(alarm ? new THREE.Color(statusColor(pal, worst)) : warm);
    if (light.current) {
      light.current.color.lerp(col, 0.2);
      light.current.intensity = alarm ? 6 + 10 * pulse : 5;
    }
    if (strip.current) strip.current.color.lerp(alarm ? col : warm, 0.2);
  });
  return (
    <group key={id}>
      <pointLight ref={light} position={m.pos} distance={m.r * 4.5} decay={1.6} intensity={5} />
      <group position={m.pos} rotation={propRotation(m.axis)}>
        <mesh position={[0, -m.r * 0.62, m.r * 0.55]}>
          <boxGeometry args={[m.len * 0.8, 0.04, 0.08]} />
          <meshBasicMaterial ref={strip} color="#fff3df" />
        </mesh>
      </group>
    </group>
  );
}

/* ------------------------------------------------------------------ Wi-Fi: access point, receiver, wavefronts */
function apPosition(m: Mod3D): THREE.Vector3 {
  const p = new THREE.Vector3(...m.pos);
  if (m.axis === "x") p.add(new THREE.Vector3(-m.len / 2 + 0.6, m.r - 0.25, 0));
  else p.add(new THREE.Vector3(0, m.r - 0.25, (m.pos[2] < 0 ? -1 : 1) * (m.len / 2 - 0.5)));
  return p;
}
function rxPosition(m: Mod3D): THREE.Vector3 {
  const p = new THREE.Vector3(...m.pos);
  if (m.axis === "x") p.add(new THREE.Vector3(m.len / 2 - 0.6, -m.r + 0.35, 0));
  else p.add(new THREE.Vector3(0, -m.r + 0.35, (m.pos[2] < 0 ? 1 : -1) * (m.len / 2 - 0.6)));
  return p;
}

function AccessPoint({ m, pal, active, still }: { m: Mod3D; pal: Palette; active: boolean; still: boolean }) {
  const ap = useMemo(() => apPosition(m), [m]);
  const rx = useMemo(() => rxPosition(m), [m]);
  const shells = useRef<(THREE.Mesh | null)[]>([]);
  const led = useRef<THREE.MeshBasicMaterial>(null);
  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    shells.current.forEach((s, k) => {
      if (!s) return;
      const ph = still ? 0.5 : (t * 0.5 + k / 3) % 1;
      s.scale.setScalar(0.15 + ph * m.r * 1.3);
      (s.material as THREE.MeshBasicMaterial).opacity = active ? (1 - ph) * 0.14 : 0;
    });
    if (led.current) led.current.opacity = active ? 0.55 + 0.45 * Math.sin(t * 6) : 0.25;
  });
  return (
    <group>
      <group position={ap}>
        <mesh>
          <boxGeometry args={[0.46, 0.1, 0.3]} />
          <meshStandardMaterial color={pal.panel} roughness={0.4} />
        </mesh>
        {[-0.14, 0.14].map((x) => (
          <mesh key={x} position={[x, 0.16, 0]}>
            <cylinderGeometry args={[0.015, 0.015, 0.24, 6]} />
            <meshStandardMaterial color="#3b4458" />
          </mesh>
        ))}
        <mesh position={[0.14, -0.06, 0.12]}>
          <sphereGeometry args={[0.035, 8, 8]} />
          <meshBasicMaterial ref={led} color={pal.accent} transparent />
        </mesh>
        {[0, 1, 2].map((k) => (
          <mesh key={k} ref={(el) => (shells.current[k] = el)}>
            <sphereGeometry args={[1, 32, 16]} />
            <meshBasicMaterial color={pal.accent} transparent opacity={0.1} depthWrite={false} side={THREE.BackSide} />
          </mesh>
        ))}
      </group>
      <mesh position={rx}>
        <boxGeometry args={[0.32, 0.08, 0.24]} />
        <meshStandardMaterial color={pal.panel} />
      </mesh>
    </group>
  );
}

/* ------------------------------------------------------------------ signal pulses: direct path + two wall bounces */
const BEADS = 7, PATHS = 3;
function SignalBeads({ m, anchor, strength, pal, still }: {
  m: Mod3D; anchor: React.RefObject<THREE.Vector3>; strength: () => number; pal: Palette; still: boolean;
}) {
  const ap = useMemo(() => apPosition(m), [m]);
  const rx = useMemo(() => rxPosition(m), [m]);
  const inst = useRef<THREE.InstancedMesh>(null);
  const mat = useRef<THREE.MeshBasicMaterial>(null);
  const line = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints([ap, new THREE.Vector3(), rx]);
    return new THREE.Line(g, new THREE.LineDashedMaterial({ color: pal.accent, dashSize: 0.14, gapSize: 0.1, transparent: true, opacity: 0.5 }));
  }, [ap, rx, pal.accent]);
  const tmp = useMemo(() => ({ o: new THREE.Object3D(), a: new THREE.Vector3(), b: new THREE.Vector3(), c: new THREE.Vector3(), up: new THREE.Vector3(), side: new THREE.Vector3() }), []);

  useFrame(({ clock }) => {
    const s = strength();
    const body = anchor.current;
    // line through the body
    const pos = line.geometry.attributes.position as THREE.BufferAttribute;
    pos.setXYZ(1, body.x, body.y, body.z);
    pos.needsUpdate = true;
    line.computeLineDistances();
    const lm = line.material as THREE.LineDashedMaterial;
    lm.opacity = 0.1 + 0.5 * s;
    (lm as unknown as { dashOffset: number }).dashOffset = -clock.elapsedTime * 0.6;

    if (!inst.current || !mat.current) return;
    mat.current.opacity = 0.2 + 0.8 * s;
    inst.current.visible = s > 0.02;
    // reflection points on the ceiling and floor beside the crew member
    const centre = tmp.c.set(...m.pos);
    if (m.axis === "x") { centre.x = body.x; tmp.side.set(0, 0, 1); } else { centre.z = body.z; tmp.side.set(1, 0, 0); }
    tmp.up.set(0, 1, 0);
    const wallA = tmp.a.copy(centre).addScaledVector(tmp.up, m.r * 0.9).addScaledVector(tmp.side, -m.r * 0.35);
    const wallB = tmp.b.copy(centre).addScaledVector(tmp.up, -m.r * 0.6).addScaledVector(tmp.side, m.r * 0.5);
    const paths = [[ap, body, rx], [ap, wallA, body, rx], [ap, body, wallB, rx]];
    const t = still ? 0 : clock.elapsedTime * 0.35;
    let i = 0;
    for (const p of paths) {
      const lens = p.slice(1).map((q, k) => q.distanceTo(p[k]));
      const total = lens.reduce((x, y) => x + y, 0) || 1;
      for (let k = 0; k < BEADS; k++) {
        let d = ((t + k / BEADS) % 1) * total, seg = 0;
        while (seg < lens.length - 1 && d > lens[seg]) { d -= lens[seg]; seg++; }
        const f = Math.min(1, d / (lens[seg] || 1));
        tmp.o.position.lerpVectors(p[seg], p[seg + 1], f);
        const sc = (i % BEADS === 0 ? 1.3 : 1) * (p === paths[0] ? 1 : 0.75);
        tmp.o.scale.setScalar(sc);
        tmp.o.updateMatrix();
        inst.current.setMatrixAt(i++, tmp.o.matrix);
      }
    }
    inst.current.instanceMatrix.needsUpdate = true;
  });
  return (
    <group>
      <primitive object={line} />
      <instancedMesh ref={inst} args={[undefined, undefined, BEADS * PATHS]} frustumCulled={false}>
        <sphereGeometry args={[0.045, 10, 8]} />
        <meshBasicMaterial ref={mat} color={pal.accent} transparent depthWrite={false} />
      </instancedMesh>
    </group>
  );
}

/* ------------------------------------------------------------------ astronaut */
interface AstroProps {
  id: string;
  home: THREE.Vector3;
  axis: Axis;
  module: Mod3D;
  framesRef: React.RefObject<Record<string, Frame>>;
  history: React.RefObject<Record<string, CrewHistory>>;
  lastFrameAt: React.RefObject<number>;
  speedRef: React.RefObject<number>;
  anchor: React.RefObject<THREE.Vector3>;
  pal: Palette;
  selected: boolean;
  reduceMotion: boolean;
  onSelect: (id: string) => void;
}

const Limb = forwardRef<THREE.Group, { len: number; r: number; color: string; position: [number, number, number] }>(
  function Limb({ len, r, color, position }, ref) {
    return (
      <group ref={ref} position={position}>
        <mesh position={[0, -len / 2, 0]}>
          <capsuleGeometry args={[r, len, 4, 10]} />
          <meshStandardMaterial color={color} roughness={0.7} />
        </mesh>
      </group>
    );
  });

function Astronaut({ id, home, axis, module, framesRef, history, lastFrameAt, speedRef, anchor, pal, selected, reduceMotion, onSelect }: AstroProps) {
  const root = useRef<THREE.Group>(null);
  const body = useRef<THREE.Group>(null);
  const chest = useRef<THREE.Group>(null);
  const bag = useRef<THREE.Group>(null);
  const figure = useRef<THREE.Group>(null);
  const ring = useRef<THREE.Mesh>(null);
  const ringMat = useRef<THREE.MeshStandardMaterial>(null);
  const halo = useRef<THREE.MeshBasicMaterial>(null);
  const legL = useRef<THREE.Group>(null), legR = useRef<THREE.Group>(null);
  const armL = useRef<THREE.Group>(null), armR = useRef<THREE.Group>(null);
  const tumble = useRef(0);
  const patchTex = useMemo(missionPatch, []);
  const seed = useMemo(() => [...id].reduce((a, c) => a + c.charCodeAt(0), 0) % 17, [id]);
  const tmp = useMemo(() => new THREE.Vector3(), []);
  const col = useMemo(() => new THREE.Color(), []);

  useFrame(({ clock }, dt) => {
    const f = framesRef.current[id];
    if (!root.current || !body.current) return;
    const t = clock.elapsedTime + seed;
    const mode = !f ? "idle" : f.activity === "fall" ? "fall" : f.context === "SLEEP" ? "sleep" :
      f.activity === "lie_down" && f.status !== "NOMINAL" ? "still" : f.activity === "run" || f.context === "EXERCISE" ? "run" :
      f.activity === "walk" ? "walk" : f.activity === "sit_down" || f.activity === "pickup" ? "work" : "float";
    const drift = reduceMotion ? 0 : 1;

    // where the body is: drifting in microgravity, strapped in a bag, on the treadmill, or translating along the lab
    let x = home.x, y = home.y, z = home.z;
    if (mode === "run") { y = floorY(module) + 1.05; }
    else if (mode === "fall") { y = THREE.MathUtils.lerp(root.current.position.y, floorY(module) + 0.45, 0.04); }
    else if (mode === "still") { y = floorY(module) + 0.35; }
    else if (mode !== "sleep") {
      const along = mode === "walk" ? Math.sin(t * 0.35) * 2.2 : Math.sin(t * 0.21) * 0.35;
      const off = axis === "x" ? tmp.set(along, 0, Math.sin(t * 0.17) * 0.25) : tmp.set(Math.sin(t * 0.17) * 0.2, 0, along * 0.3);
      x += off.x * drift; z += off.z * drift;
      y += Math.sin(t * 0.6) * 0.14 * drift;
    }
    if (mode === "run") y += Math.abs(Math.sin(t * 9)) * 0.06 * drift;
    root.current.position.set(x, y, z);

    // orientation
    if (mode === "fall") tumble.current += dt * 4.5;
    else tumble.current *= 0.9;
    const targetZ = mode === "still" ? Math.PI / 2 : mode === "sleep" || mode === "run" ? 0 : Math.sin(t * 0.3) * 0.15 * drift;
    body.current.rotation.z = THREE.MathUtils.lerp(body.current.rotation.z, targetZ, 0.06) + (mode === "fall" ? Math.sin(tumble.current) * 0.9 : 0);
    body.current.rotation.x = mode === "fall" ? tumble.current : THREE.MathUtils.lerp(body.current.rotation.x % (Math.PI * 2), 0, 0.08);
    body.current.rotation.y = mode === "walk" ? (Math.cos(t * 0.35) > 0 ? 0 : Math.PI) + (axis === "x" ? Math.PI / 2 : 0)
      : mode === "run" ? -Math.PI / 2 : mode === "work" ? Math.PI : Math.sin(t * 0.11) * 0.4 * drift;

    // breathing: chest / bag expand with the replayed CSI breathing waveform
    const w = liveWaveSample(history.current[id], lastFrameAt.current, speedRef.current);
    const breathe = 1 + 0.08 * w;
    if (chest.current) chest.current.scale.set(breathe, 1, breathe);
    if (bag.current) {
      bag.current.visible = mode === "sleep";
      bag.current.scale.set(breathe, 1, breathe);
    }
    if (figure.current) figure.current.visible = mode !== "sleep";

    // limbs
    const swing = mode === "run" ? Math.sin(t * 9) * 0.9 : mode === "walk" ? Math.sin(t * 3) * 0.35 : 0;
    const legBase = mode === "run" ? 0 : mode === "work" ? 1.2 : 0.45;
    if (legL.current) legL.current.rotation.x = legBase + swing;
    if (legR.current) legR.current.rotation.x = legBase - swing;
    const armBase = mode === "work" ? -1.2 : mode === "run" ? -0.2 : -0.7;
    if (armL.current) armL.current.rotation.set(armBase - swing * 0.6, 0, mode === "run" ? 0.15 : 0.5);
    if (armR.current) armR.current.rotation.set(armBase + swing * 0.6, 0, mode === "run" ? -0.15 : -0.5);

    // status ring and halo
    const status = f?.status ?? "NOMINAL";
    const alarm = status === "WARNING" || status === "EMERGENCY";
    col.set(statusColor(pal, status));
    if (ringMat.current) {
      ringMat.current.color.copy(col);
      ringMat.current.emissive.copy(col);
      ringMat.current.emissiveIntensity = alarm ? 0.6 + 0.6 * Math.sin(clock.elapsedTime * 7) : 0.25;
    }
    if (ring.current) ring.current.scale.setScalar(alarm && !reduceMotion ? 1 + 0.15 * Math.sin(clock.elapsedTime * 7) : 1);
    if (halo.current) {
      halo.current.color.copy(col);
      halo.current.opacity = alarm ? 0.18 + 0.12 * Math.sin(clock.elapsedTime * 7) : selected ? 0.1 : 0;
    }
    root.current.getWorldPosition(anchor.current);
    anchor.current.y += 0.15;
  });

  const over = (e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); document.body.style.cursor = "pointer"; };
  const out = () => { document.body.style.cursor = ""; };
  return (
    <group ref={root} onClick={(e) => { e.stopPropagation(); onSelect(id); }} onPointerOver={over} onPointerOut={out}>
      <mesh ref={ring} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.62, selected ? 0.05 : 0.03, 8, 48]} />
        <meshStandardMaterial ref={ringMat} color={pal.nominal} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.95, 24, 16]} />
        <meshBasicMaterial ref={halo} color={pal.nominal} transparent opacity={0} depthWrite={false} />
      </mesh>
      <group ref={body}>
        {/* sleeping bag, strapped vertically to the crew-quarters wall */}
        <group ref={bag}>
          <mesh>
            <capsuleGeometry args={[0.3, 0.95, 6, 16]} />
            <meshStandardMaterial color={pal.bag} roughness={0.9} />
          </mesh>
          <mesh position={[0, 0.78, 0.02]}>
            <sphereGeometry args={[0.19, 20, 16]} />
            <meshStandardMaterial color={pal.suit} roughness={0.6} />
          </mesh>
        </group>
        <group ref={figure}>
          <group ref={chest}>
            <mesh>
              <capsuleGeometry args={[0.2, 0.42, 6, 14]} />
              <meshStandardMaterial color={pal.shirt} roughness={0.65} />
            </mesh>
          </group>
          <mesh position={[0, 0.52, 0]}>
            <sphereGeometry args={[0.17, 20, 16]} />
            <meshStandardMaterial color={pal.suit} roughness={0.5} />
          </mesh>
          <Limb ref={armL} len={0.42} r={0.06} color={pal.suit} position={[0.26, 0.25, 0]} />
          <Limb ref={armR} len={0.42} r={0.06} color={pal.suit} position={[-0.26, 0.25, 0]} />
          {/* polo collar and shoulder patch */}
          <mesh position={[0, 0.36, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.1, 0.03, 8, 24]} />
            <meshStandardMaterial color={pal.shirt} roughness={0.7} />
          </mesh>
          <mesh position={[0.205, 0.22, 0]} rotation={[0, Math.PI / 2, 0]}>
            <circleGeometry args={[0.07, 24]} />
            <meshStandardMaterial map={patchTex} transparent />
          </mesh>
          <mesh position={[0, -0.26, 0]}>
            <cylinderGeometry args={[0.205, 0.205, 0.05, 20]} />
            <meshStandardMaterial color="#2f3440" />
          </mesh>
          <Limb ref={legL} len={0.55} r={0.075} color="#4f5a6e" position={[0.11, -0.32, 0]} />
          <Limb ref={legR} len={0.55} r={0.075} color="#4f5a6e" position={[-0.11, -0.32, 0]} />
        </group>
      </group>
    </group>
  );
}

/* ------------------------------------------------------------------ camera controls + focus */
const STATION_CENTER = new THREE.Vector3(-0.4, -0.3, 0.6);
const VIEW_DIR = new THREE.Vector3(0.08, 0.55, 1).normalize();
const STATION_HALF_WIDTH = 14.2;
const STATION_HALF_HEIGHT = 5.5;

function Controls({ focus, anchors, autoRotate, resetKey, walk }: {
  focus: string | null; anchors: Record<string, React.RefObject<THREE.Vector3>>; autoRotate: boolean; resetKey: number; walk: boolean;
}) {
  const { camera, gl, size } = useThree();
  const ctl = useRef<OrbitControls | null>(null);
  const touched = useRef(false);
  // frame the whole station for the current aspect ratio (until the user takes over the camera)
  useEffect(() => {
    touched.current = false;
  }, [resetKey]);
  useEffect(() => {
    if (touched.current || walk) return;
    const cam = camera as THREE.PerspectiveCamera;
    const vfov = THREE.MathUtils.degToRad(cam.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * (size.width / size.height));
    const dist = Math.max(STATION_HALF_WIDTH / Math.tan(hfov / 2), STATION_HALF_HEIGHT / Math.tan(vfov / 2));
    cam.position.copy(STATION_CENTER).addScaledVector(VIEW_DIR, dist);
    cam.lookAt(STATION_CENTER);
    ctl.current?.target.copy(STATION_CENTER);
  }, [camera, size.width, size.height, resetKey, walk]);
  useEffect(() => {
    const c = new OrbitControls(camera, gl.domElement);
    c.addEventListener("start", () => (touched.current = true));
    c.target.copy(STATION_CENTER);
    c.enableDamping = true;
    c.dampingFactor = 0.08;
    c.minDistance = 4;
    c.maxDistance = 70;
    c.maxPolarAngle = Math.PI * 0.85;
    c.autoRotateSpeed = 0.5;
    // A plain mouse wheel scrolls the page; Ctrl + wheel (or a trackpad pinch) zooms the station.
    c.enableZoom = false;
    const host = gl.domElement.parentElement ?? gl.domElement;
    const onWheel = (e: WheelEvent) => { c.enableZoom = e.ctrlKey || e.metaKey; };
    host.addEventListener("wheel", onWheel, { capture: true, passive: true });
    ctl.current = c;
    return () => { host.removeEventListener("wheel", onWheel, { capture: true }); c.dispose(); };
  }, [camera, gl]);
  useFrame(() => {
    const c = ctl.current;
    if (!c) return;
    c.enabled = !walk; // the walk-through takes over the mouse and camera
    if (walk) return;
    c.autoRotate = autoRotate;
    const target = focus && anchors[focus] ? anchors[focus].current : STATION_CENTER;
    c.target.lerp(target, 0.06);
    c.update();
  });
  return null;
}

/* ------------------------------------------------------------------ walk-through: fly inside the modules in first person */
const WALL = 0.28; // stay this far from the walls
const END_OVERLAP = 0.6; // lets you pass between modules that meet end to end

/** True while a point is inside the pressurised volume (the modules and the connectors between them). */
function insideStation(p: THREE.Vector3): boolean {
  for (const m of Object.values(MODULES_3D)) {
    const along = m.axis === "x" ? p.x - m.pos[0] : p.z - m.pos[2];
    if (Math.abs(along) > m.len / 2 + END_OVERLAP) continue;
    const across = m.axis === "x" ? p.z - m.pos[2] : p.x - m.pos[0];
    if (Math.hypot(p.y - m.pos[1], across) <= m.r - WALL) return true;
  }
  for (const c of CONNECTORS) {
    const lo = Math.min(c.from[0], c.to[0]) - 0.1, hi = Math.max(c.from[0], c.to[0]) + 0.1;
    if (p.x >= lo && p.x <= hi && Math.hypot(p.y - c.from[1], p.z - c.from[2]) <= c.r - WALL) return true;
  }
  return false;
}

const WALK_KEYS: Record<string, string> = {
  KeyW: "fwd", ArrowUp: "fwd", KeyS: "back", ArrowDown: "back", KeyA: "left", ArrowLeft: "left",
  KeyD: "right", ArrowRight: "right", KeyR: "up", KeyE: "up", KeyF: "down", KeyQ: "down",
  ShiftLeft: "fast", ShiftRight: "fast",
};

function WalkControls({ active, onExit }: { active: boolean; onExit?: () => void }) {
  const { camera, gl } = useThree();
  const held = useRef(new Set<string>());
  const look = useRef({ yaw: -Math.PI / 2, pitch: -0.05 });
  const tmp = useMemo(() => ({ dir: new THREE.Vector3(), right: new THREE.Vector3(), move: new THREE.Vector3(), next: new THREE.Vector3() }), []);

  useEffect(() => {
    if (!active) return;
    const cam = camera as THREE.PerspectiveCamera;
    const before = { fov: cam.fov, near: cam.near };
    cam.fov = 72;
    cam.near = 0.05;
    cam.updateProjectionMatrix();
    cam.position.set(-10, -0.2, 0); // the crew hub in Node 2, where nobody floats, looking down the whole station
    look.current = { yaw: -Math.PI / 2, pitch: -0.05 };

    const canvas = gl.domElement;
    canvas.style.cursor = "grab";
    canvas.style.touchAction = "none";
    let dragging = false, lx = 0, ly = 0;
    const down = (e: PointerEvent) => { dragging = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = "grabbing"; };
    const move = (e: PointerEvent) => {
      if (!dragging) return;
      look.current.yaw -= (e.clientX - lx) * 0.0045;
      look.current.pitch = THREE.MathUtils.clamp(look.current.pitch - (e.clientY - ly) * 0.0045, -1.45, 1.45);
      lx = e.clientX; ly = e.clientY;
    };
    const up = (e: PointerEvent) => { dragging = false; if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId); canvas.style.cursor = "grab"; };
    const typing = (e: KeyboardEvent) => ["INPUT", "SELECT", "TEXTAREA"].includes((e.target as HTMLElement | null)?.tagName ?? "");
    const keyDown = (e: KeyboardEvent) => {
      if (typing(e)) return;
      if (e.code === "Escape") { onExit?.(); return; }
      const k = WALK_KEYS[e.code];
      if (k) { held.current.add(k); e.preventDefault(); }
    };
    const keyUp = (e: KeyboardEvent) => { const k = WALK_KEYS[e.code]; if (k) held.current.delete(k); };
    const clear = () => held.current.clear();

    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    window.addEventListener("blur", clear);
    return () => {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", clear);
      held.current.clear();
      canvas.style.cursor = "";
      canvas.style.touchAction = "";
      cam.fov = before.fov;
      cam.near = before.near;
      cam.updateProjectionMatrix();
    };
  }, [active, camera, gl, onExit]);

  useFrame((_, dt) => {
    if (!active) return;
    camera.rotation.set(look.current.pitch, look.current.yaw, 0, "YXZ");
    const h = held.current;
    const f = (h.has("fwd") ? 1 : 0) - (h.has("back") ? 1 : 0);
    const s = (h.has("right") ? 1 : 0) - (h.has("left") ? 1 : 0);
    const v = (h.has("up") ? 1 : 0) - (h.has("down") ? 1 : 0);
    if (!f && !s && !v) return;
    tmp.dir.set(0, 0, -1).applyEuler(camera.rotation);
    tmp.right.set(1, 0, 0).applyEuler(camera.rotation);
    tmp.move.set(0, v, 0).addScaledVector(tmp.dir, f).addScaledVector(tmp.right, s);
    tmp.move.multiplyScalar(Math.min(dt, 0.05) * (h.has("fast") ? 5 : 2.6));
    // try the whole step first, then slide along each axis so the walls never stop you dead
    const p = camera.position;
    tmp.next.copy(p).add(tmp.move);
    if (insideStation(tmp.next)) { p.copy(tmp.next); return; }
    for (const axis of ["x", "y", "z"] as const) {
      tmp.next.copy(p);
      tmp.next[axis] += tmp.move[axis];
      if (insideStation(tmp.next)) p.copy(tmp.next);
    }
  });
  return null;
}

/* ------------------------------------------------------------------ label projection (HTML overlay) */
function LabelProjector({ anchors, labelEls }: {
  anchors: Record<string, React.RefObject<THREE.Vector3>>;
  labelEls: React.RefObject<Record<string, HTMLDivElement | null>>;
}) {
  const { camera, size } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    for (const [id, a] of Object.entries(anchors)) {
      const el = labelEls.current[id];
      if (!el) continue;
      v.copy(a.current);
      v.y += 0.95;
      v.project(camera);
      const hidden = v.z > 1;
      el.style.transform = `translate(-50%, -100%) translate(${((v.x + 1) / 2) * size.width}px, ${((1 - v.y) / 2) * size.height}px)`;
      el.style.opacity = hidden ? "0" : "1";
    }
  });
  return null;
}

/* ------------------------------------------------------------------ scene root */
export interface StationProps {
  crew: { id: string; module: string }[];
  frames: Record<string, Frame>;
  framesRef: React.RefObject<Record<string, Frame>>;
  history: React.RefObject<Record<string, CrewHistory>>;
  lastFrameAt: React.RefObject<number>;
  speedRef: React.RefObject<number>;
  running: boolean;
  selected: string | null;
  onSelect: (id: string | null) => void;
  theme: string;
  labelEls: React.RefObject<Record<string, HTMLDivElement | null>>;
  autoRotate?: boolean;
  resetKey?: number;
  /** First-person walk-through inside the modules; Escape calls onExitWalk. */
  walk?: boolean;
  onExitWalk?: () => void;
}

export function homePosition(module: string, k: number, n: number): { pos: THREE.Vector3; axis: Axis } {
  const m = MODULES_3D[module] ?? MODULES_3D.LAB;
  const p = new THREE.Vector3(...m.pos);
  const spread = (k - (n - 1) / 2) * 1.6;
  if (m.axis === "x") p.x += spread; else p.z += spread * 0.5;
  return { pos: p, axis: m.axis };
}

export function Station3D(props: StationProps) {
  const [box, inView] = useInView<HTMLDivElement>();
  const { crew, frames, framesRef, history, lastFrameAt, speedRef, running, selected, onSelect, theme, labelEls } = props;
  const pal = useMemo(readPalette, [theme]);
  const reduceMotion = useMemo(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches, []);
  const anchors = useMemo(() => Object.fromEntries(crew.map((c) => [c.id, { current: new THREE.Vector3() }])), [crew]) as
    Record<string, React.RefObject<THREE.Vector3>>;

  // where is each crew member right now (modules can change via the scenario)
  const placement = useMemo(() => {
    const byMod: Record<string, string[]> = {};
    crew.forEach((c) => (byMod[frames[c.id]?.module ?? c.module] ??= []).push(c.id));
    const out: Record<string, { pos: THREE.Vector3; axis: Axis; module: string }> = {};
    for (const [mod, ids] of Object.entries(byMod)) ids.forEach((id, k) => (out[id] = { ...homePosition(mod, k, ids.length), module: mod }));
    return out;
  }, [crew, frames]);
  const placementRef = useRef(placement);
  placementRef.current = placement;

  const occupied = new Set(Object.values(placement).map((p) => p.module));

  return (
    <div ref={box} className="absolute inset-0">
    <Canvas frameloop={inView ? "always" : "never"} camera={{ position: [3, 9.5, 17], fov: 38, far: 600 }} dpr={[1, 1.5]} onPointerMissed={() => onSelect(null)}
      aria-label="3D view of the station with crew">
      <color attach="background" args={["#050a17"]} />
      <ambientLight intensity={0.35} />
      <hemisphereLight args={["#cfe0ff", "#1b3566", 0.7]} />
      <directionalLight position={[14, 16, 8]} intensity={2.2} color="#fff8ee" />
      <directionalLight position={[-10, -6, -8]} intensity={0.35} color="#7fa8ff" />

      <Stars />
      <Earth still={reduceMotion} />
      <Exterior still={reduceMotion} />

      {Object.entries(MODULES_3D).map(([id, m]) => (
        <group key={id}>
          <Hull m={m} pal={pal} />
          <Racks m={m} pal={pal} />
          <ModuleLight id={id} m={m} pal={pal} framesRef={framesRef} still={reduceMotion}
            occupants={() => Object.entries(placementRef.current).filter(([, p]) => p.module === id).map(([c]) => c)} />
        </group>
      ))}
      {CONNECTORS.map((c, k) => <Connector key={k} c={c} pal={pal} />)}
      <Treadmill m={MODULES_3D.NODE1} />
      <Workbench m={MODULES_3D.LAB} />
      {Object.entries(MODULES_3D).filter(([id]) => id !== "NODE2").map(([id, m]) => (
        <AccessPoint key={id} m={m} pal={pal} active={running && occupied.has(id)} still={reduceMotion} />
      ))}

      {crew.map((c) => {
        const pl = placement[c.id];
        if (!pl || !frames[c.id]) return null;
        const m = MODULES_3D[pl.module] ?? MODULES_3D.LAB;
        return (
          <group key={c.id}>
            <Astronaut
              id={c.id} home={pl.pos} axis={pl.axis} module={m} framesRef={framesRef} history={history} lastFrameAt={lastFrameAt}
              speedRef={speedRef} anchor={anchors[c.id]} pal={pal} selected={selected === c.id} reduceMotion={reduceMotion}
              onSelect={onSelect}
            />
            <SignalBeads
              m={m} anchor={anchors[c.id]} pal={pal} still={reduceMotion || !running}
              strength={() => {
                const fr = framesRef.current[c.id];
                if (!fr) return 0;
                return fr.br_available !== false && fr.bpm != null ? fr.br_conf : Math.min(1, 0.4 + fr.motion * 8);
              }}
            />
          </group>
        );
      })}
      <Controls focus={selected} anchors={anchors} autoRotate={!!props.autoRotate && !reduceMotion} resetKey={props.resetKey ?? 0} walk={!!props.walk} />
      <WalkControls active={!!props.walk} onExit={props.onExitWalk} />
      <LabelProjector anchors={anchors} labelEls={labelEls} />
    </Canvas>
    </div>
  );
}
