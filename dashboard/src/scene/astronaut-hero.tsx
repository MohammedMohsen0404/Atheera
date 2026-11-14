"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useInView } from "./in-view";
import { missionPatch } from "./patch";
import { liveWaveSample, type CrewHistory } from "@/lib/telemetry";
import type { Frame } from "@/lib/types";

/* A sculpted, translucent crew member. The lungs inside the chest expand and contract with the
   breathing signal recovered from Wi-Fi CSI; nothing here is animated from a made-up rhythm.
   Callout anchors are projected to screen space for the HTML labels around the figure. */

export type AnchorName = "chest" | "head" | "hand" | "router";

interface Props {
  crewId: string | null;
  framesRef: React.RefObject<Record<string, Frame>>;
  history: React.RefObject<Record<string, CrewHistory>>;
  lastFrameAt: React.RefObject<number>;
  speedRef: React.RefObject<number>;
  calloutEls: React.RefObject<Record<AnchorName, HTMLDivElement | null>>;
  theme: string;
}

function readColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string, fb: string) => cs.getPropertyValue(n).trim() || fb;
  return {
    body: v("--figure", "#f3f5f2"), shirt: "#0a4aa6", rim: v("--brand-2", "#4f8065"),
    lung: v("--lung", "#d6766a"), lungHi: v("--lung-hi", "#f1a99b"), pants: "#4f5a6e", pocket: "#46505f", shoe: "#262c38", belt: "#2f3440",
    router: v("--figure", "#f5f7f4"), emergency: v("--emergency", "#d6453d"),
  };
}

/* ---------------------------------------------------------------- geometry helpers */
const lathe = (pts: [number, number][], seg = 40) => new THREE.LatheGeometry(pts.map(([r, y]) => new THREE.Vector2(r, y)), seg);

/** A tapered limb hanging down from its joint (y = 0 at the joint). */
function limbGeometry(len: number, r0: number, r1: number) {
  return lathe([[0, 0.02], [r0 * 0.9, 0.012], [r0, -len * 0.12], [(r0 + r1) / 2 * 1.04, -len * 0.5], [r1, -len * 0.9], [r1 * 0.85, -len], [0, -len - 0.01]], 28);
}

/** Torso silhouette: hips, waist, ribcage, shoulders. Flattened front-to-back afterwards. */
const TORSO = lathe([
  [0, -0.07], [0.17, -0.06], [0.22, 0.0], [0.215, 0.1], [0.19, 0.22], [0.205, 0.34],
  [0.235, 0.47], [0.255, 0.58], [0.25, 0.65], [0.2, 0.71], [0.1, 0.75], [0.065, 0.765], [0, 0.765],
], 48);

/** Lung lobe: an egg that tapers to an apex, with the inner (heart-side) face flattened. */
function lungGeometry(side: 1 | -1) {
  const g = new THREE.SphereGeometry(0.1, 40, 32);
  const p = g.attributes.position as THREE.BufferAttribute;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    v.y *= 1.75;
    const up = Math.max(0, v.y) / 0.175;
    v.x *= 1 - 0.45 * up;
    v.z *= (1 - 0.35 * up) * 0.75;
    if (v.x * side < 0) v.x *= 0.45; // flat medial face
    if (v.y < -0.12) v.y = -0.12 + (v.y + 0.12) * 0.4; // flattened base (diaphragm)
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}

/* ---------------------------------------------------------------- figure */
function Figure({ crewId, framesRef, history, lastFrameAt, speedRef, anchors, colors }: {
  crewId: string | null; framesRef: Props["framesRef"]; history: Props["history"]; lastFrameAt: Props["lastFrameAt"];
  speedRef: Props["speedRef"]; anchors: Record<AnchorName, THREE.Object3D | null>; colors: ReturnType<typeof readColors>;
}) {
  const root = useRef<THREE.Group>(null);
  const body = useRef<THREE.Group>(null);
  const lungs = useRef<THREE.Group>(null);
  const torso = useRef<THREE.Mesh>(null);
  const armL = useRef<THREE.Group>(null), armR = useRef<THREE.Group>(null);
  const foreL = useRef<THREE.Group>(null), foreR = useRef<THREE.Group>(null);
  const legL = useRef<THREE.Group>(null), legR = useRef<THREE.Group>(null);
  const shinL = useRef<THREE.Group>(null), shinR = useRef<THREE.Group>(null);
  const shirtMat = useRef<THREE.MeshPhysicalMaterial>(null);
  const tumble = useRef(0);
  const patchTex = useMemo(missionPatch, []);
  const shirtColor = useMemo(() => new THREE.Color(), []);

  const geo = useMemo(() => ({
    upperArm: limbGeometry(0.3, 0.062, 0.05), foreArm: limbGeometry(0.27, 0.048, 0.036),
    thigh: limbGeometry(0.42, 0.095, 0.066), shin: limbGeometry(0.4, 0.064, 0.042),
    lungL: lungGeometry(1), lungR: lungGeometry(-1),
  }), []);

  useFrame(({ clock }, dt) => {
    const t = clock.elapsedTime;
    const f = crewId ? framesRef.current[crewId] : undefined;
    if (!root.current || !body.current) return;
    const mode = !f ? "float" : f.activity === "fall" ? "fall" : f.context === "SLEEP" ? "sleep" :
      f.activity === "run" ? "run" : f.activity === "walk" ? "walk" : "float";

    // microgravity: a slow drift and turn
    root.current.position.y = Math.sin(t * 0.7) * 0.04;
    root.current.rotation.y = -0.35 + Math.sin(t * 0.22) * 0.3;
    if (mode === "fall") tumble.current += dt * 3.2; else tumble.current *= 0.92;
    body.current.rotation.z = Math.sin(tumble.current) * 0.8;
    body.current.rotation.x = mode === "fall" ? Math.sin(tumble.current * 0.7) * 0.6
      : THREE.MathUtils.lerp(body.current.rotation.x, mode === "sleep" ? 0.1 : 0.02, 0.05);

    // breathing from the replayed CSI waveform (rooms without a breathing sensor stay still)
    const w = crewId ? liveWaveSample(history.current[crewId], lastFrameAt.current, speedRef.current) : 0;
    lungs.current?.scale.set(1 + 0.1 * w, 1 + 0.045 * w, 1 + 0.1 * w);
    torso.current?.scale.set(1 + 0.018 * w, 1, 0.66 * (1 + 0.04 * w));

    // neutral body posture: arms float forward and out, knees bend; more swing when moving
    const swing = mode === "run" ? Math.sin(t * 8) * 0.8 : mode === "walk" ? Math.sin(t * 3) * 0.3 : Math.sin(t * 0.8) * 0.04;
    const relaxed = mode === "sleep" ? 0.15 : 0.32;
    armL.current?.rotation.set(-0.3 - swing * 0.5, 0, -relaxed);
    armR.current?.rotation.set(-0.3 + swing * 0.5, 0, relaxed);
    foreL.current?.rotation.set(-0.55, 0, 0);
    foreR.current?.rotation.set(-0.55, 0, 0);
    legL.current?.rotation.set(-0.35 - swing, 0, -0.05);
    legR.current?.rotation.set(-0.25 + swing, 0, 0.05);
    shinL.current?.rotation.set(0.7, 0, 0);
    shinR.current?.rotation.set(0.55, 0, 0);

    if (shirtMat.current) {
      const alarm = f && (f.status === "EMERGENCY" || f.status === "WARNING");
      shirtColor.set(alarm ? colors.emergency : colors.shirt);
      shirtMat.current.color.lerp(shirtColor, 0.1);
    }
  });

  const skin = <meshPhysicalMaterial color={colors.body} roughness={0.42} clearcoat={0.35} clearcoatRoughness={0.6} />;
  const cloth = <meshPhysicalMaterial color={colors.pants} roughness={0.8} sheen={0.4} sheenColor="#8a95ab" />;
  // sleeves and collar match how the see-through shirt reads on screen, so it all looks like one garment
  const sleeve = <meshPhysicalMaterial color="#2f78c8" roughness={0.75} sheen={0.5} sheenColor="#9cc8f2" />;
  const shoe = <meshStandardMaterial color={colors.shoe} roughness={0.5} />;
  const patch = <meshStandardMaterial map={patchTex} roughness={0.6} transparent />;

  const Arm = ({ side, upper, fore }: { side: 1 | -1; upper: React.RefObject<THREE.Group | null>; fore: React.RefObject<THREE.Group | null> }) => (
    <group position={[0.265 * side, 0.665, 0]} ref={upper}>
      <mesh>{<sphereGeometry args={[0.068, 24, 18]} />}{skin}</mesh>
      <mesh geometry={geo.upperArm}>{skin}</mesh>
      {/* short polo sleeve with the crew patch on the outside */}
      <mesh position={[0, 0.0, 0]}><sphereGeometry args={[0.08, 24, 18, 0, Math.PI * 2, 0, Math.PI / 2]} />{sleeve}</mesh>
      <mesh position={[0, -0.075, 0]}><cylinderGeometry args={[0.08, 0.07, 0.15, 24, 1, true]} />{sleeve}</mesh>
      {side === 1 && (
        <mesh position={[0.08, -0.1, 0]} rotation={[0, Math.PI / 2, 0]}><circleGeometry args={[0.045, 32]} />{patch}</mesh>
      )}
      <group position={[0, -0.3, 0]} ref={fore}>
        <mesh><sphereGeometry args={[0.05, 20, 16]} />{skin}</mesh>
        <mesh geometry={geo.foreArm}>{skin}</mesh>
        <mesh position={[0, -0.31, 0.01]} scale={[0.8, 1.25, 0.5]} ref={side === 1 ? (el) => (anchors.hand = el) : undefined}>
          <sphereGeometry args={[0.05, 20, 16]} />{skin}
        </mesh>
      </group>
    </group>
  );
  const Leg = ({ side, upper, lower }: { side: 1 | -1; upper: React.RefObject<THREE.Group | null>; lower: React.RefObject<THREE.Group | null> }) => (
    <group position={[0.11 * side, -0.02, 0]} ref={upper}>
      <mesh geometry={geo.thigh}>{cloth}</mesh>
      {/* cargo pocket on the outer thigh */}
      <mesh position={[0.068 * side, -0.25, 0]} rotation={[0, 0, -0.08 * side]}>
        <boxGeometry args={[0.018, 0.11, 0.09]} /><meshStandardMaterial color={colors.pocket} roughness={0.85} />
      </mesh>
      <group position={[0, -0.42, 0]} ref={lower}>
        <mesh><sphereGeometry args={[0.066, 20, 16]} />{cloth}</mesh>
        <mesh geometry={geo.shin}>{cloth}</mesh>
        <mesh position={[0, -0.4, 0]}><cylinderGeometry args={[0.046, 0.05, 0.06, 18]} />{shoe}</mesh>
        <mesh position={[0, -0.44, 0.05]} scale={[0.8, 0.55, 1.75]}><sphereGeometry args={[0.06, 20, 16]} />{shoe}</mesh>
      </group>
    </group>
  );

  return (
    <group ref={root}>
      <group ref={body}>
        {/* head + neck */}
        <mesh position={[0, 0.97, 0]} scale={[0.92, 1.12, 0.98]}>
          <sphereGeometry args={[0.155, 48, 36]} />{skin}
        </mesh>
        <object3D position={[0.13, 1.05, 0.05]} ref={(el) => (anchors.head = el)} />
        <mesh position={[0, 0.79, 0]}>
          <cylinderGeometry args={[0.058, 0.07, 0.12, 24]} />{skin}
        </mesh>

        {/* lungs + airway sit inside a translucent shirt */}
        <group ref={lungs} position={[0, 0.47, 0.01]}>
          <mesh geometry={geo.lungL} position={[0.098, 0, 0]}>
            <meshPhysicalMaterial color={colors.lung} emissive={colors.lung} emissiveIntensity={0.28} roughness={0.38} clearcoat={0.5} />
          </mesh>
          <mesh geometry={geo.lungR} position={[-0.098, 0, 0]}>
            <meshPhysicalMaterial color={colors.lung} emissive={colors.lung} emissiveIntensity={0.28} roughness={0.38} clearcoat={0.5} />
          </mesh>
          <mesh position={[0, 0.2, 0]}>
            <cylinderGeometry args={[0.016, 0.018, 0.16, 12]} />
            <meshStandardMaterial color={colors.lungHi} />
          </mesh>
          {[1, -1].map((s) => (
            <mesh key={s} position={[0.035 * s, 0.1, 0]} rotation={[0, 0, 0.75 * s]}>
              <cylinderGeometry args={[0.011, 0.013, 0.1, 10]} />
              <meshStandardMaterial color={colors.lungHi} />
            </mesh>
          ))}
          <object3D position={[0.19, 0.0, 0.06]} ref={(el) => (anchors.chest = el)} />
        </group>
        <mesh ref={torso} geometry={TORSO} scale={[1, 1, 0.66]} renderOrder={2}>
          <meshPhysicalMaterial ref={shirtMat} color={colors.shirt} transparent opacity={0.42} roughness={0.6}
            clearcoat={0.2} depthWrite={false} side={THREE.DoubleSide} />
        </mesh>

        {/* polo collar, button placket and a chest patch on the see-through shirt */}
        <mesh position={[0, 0.755, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[1, 0.8, 1]}>
          <torusGeometry args={[0.078, 0.024, 12, 40]} />{sleeve}
        </mesh>
        <mesh position={[0, 0.68, 0.118]} rotation={[-0.45, 0, 0]}>
          <boxGeometry args={[0.035, 0.11, 0.008]} />{sleeve}
        </mesh>
        <mesh position={[0.105, 0.585, 0.17]} rotation={[-0.08, 0.25, 0]}>
          <circleGeometry args={[0.04, 32]} />{patch}
        </mesh>
        {/* belt */}
        <group position={[0, 0.0, 0]} scale={[1, 1, 0.7]}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.222, 0.02, 10, 56]} /><meshStandardMaterial color={colors.belt} roughness={0.5} />
          </mesh>
        </group>
        <mesh position={[0, 0.0, 0.158]}>
          <boxGeometry args={[0.05, 0.04, 0.012]} /><meshStandardMaterial color="#b8bfcc" metalness={0.8} roughness={0.3} />
        </mesh>

        <Arm side={-1} upper={armL} fore={foreL} />
        <Arm side={1} upper={armR} fore={foreR} />
        <mesh position={[0, -0.01, 0]} scale={[1.05, 0.55, 0.72]}>
          <sphereGeometry args={[0.215, 32, 24]} />{cloth}
        </mesh>
        <Leg side={-1} upper={legL} lower={shinL} />
        <Leg side={1} upper={legR} lower={shinR} />
      </group>
    </group>
  );
}

/* ---------------------------------------------------------------- Wi-Fi transmitter / receiver */
function Router({ x, colors, anchorRef }: { x: number; colors: ReturnType<typeof readColors>; anchorRef?: (o: THREE.Object3D | null) => void }) {
  const rings = useRef<(THREE.Mesh | null)[]>([]);
  useFrame(({ clock }) => {
    rings.current.forEach((r, k) => {
      if (!r) return;
      const ph = (clock.elapsedTime * 0.45 + k / 3) % 1;
      r.scale.setScalar(0.12 + ph * 0.85);
      (r.material as THREE.MeshBasicMaterial).opacity = (1 - ph) * 0.3;
    });
  });
  return (
    <group position={[x, 0.5, -0.25]}>
      <mesh>
        <boxGeometry args={[0.2, 0.05, 0.14]} />
        <meshPhysicalMaterial color={colors.router} roughness={0.35} clearcoat={0.6} />
      </mesh>
      {[-0.05, 0.05].map((dx) => (
        <mesh key={dx} position={[dx, 0.07, -0.04]}>
          <cylinderGeometry args={[0.008, 0.008, 0.12, 8]} />
          <meshStandardMaterial color={colors.pants} />
        </mesh>
      ))}
      <object3D position={[0, -0.05, 0]} ref={anchorRef} />
      {[0, 1, 2].map((k) => (
        <mesh key={k} ref={(el) => (rings.current[k] = el)} rotation={[0, Math.PI / 2, 0]}>
          <ringGeometry args={[0.47, 0.5, 64]} />
          <meshBasicMaterial color={colors.rim} transparent opacity={0.3} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      ))}
    </group>
  );
}

function Rays({ anchors, colors }: { anchors: Record<AnchorName, THREE.Object3D | null>; colors: ReturnType<typeof readColors> }) {
  const line = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.95, 0.5, -0.25), new THREE.Vector3(), new THREE.Vector3(0.95, 0.5, -0.25)]);
    return new THREE.Line(g, new THREE.LineDashedMaterial({ color: colors.rim, dashSize: 0.045, gapSize: 0.035, transparent: true, opacity: 0.6 }));
  }, [colors.rim]);
  const v = useMemo(() => new THREE.Vector3(), []);
  const parentInv = useMemo(() => new THREE.Matrix4(), []);
  useFrame(({ clock }) => {
    const a = anchors.chest;
    if (!a || !line.parent) return;
    a.getWorldPosition(v);
    v.applyMatrix4(parentInv.copy(line.parent.matrixWorld).invert());
    v.x -= 0.19; v.z -= 0.22; // through the middle of the chest, behind the callout anchor
    const pos = line.geometry.attributes.position as THREE.BufferAttribute;
    pos.setXYZ(1, v.x, v.y, v.z);
    pos.needsUpdate = true;
    line.computeLineDistances();
    (line.material as unknown as { dashOffset: number }).dashOffset = -clock.elapsedTime * 0.22;
  });
  return <primitive object={line} />;
}

function Projector({ anchors, els }: { anchors: Record<AnchorName, THREE.Object3D | null>; els: Props["calloutEls"] }) {
  const { camera, size } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    (Object.keys(anchors) as AnchorName[]).forEach((k) => {
      const o = anchors[k], el = els.current[k];
      if (!o || !el) return;
      o.getWorldPosition(v).project(camera);
      el.style.left = `${((v.x + 1) / 2) * size.width}px`;
      el.style.top = `${((1 - v.y) / 2) * size.height}px`;
    });
  });
  return null;
}

export function AstronautHero(p: Props) {
  const [box, inView] = useInView<HTMLDivElement>();
  const colors = useMemo(readColors, [p.theme]);
  const anchors = useMemo(() => ({ chest: null, head: null, hand: null, router: null }) as Record<AnchorName, THREE.Object3D | null>, []);
  return (
    <div ref={box} className="absolute inset-0">
    <Canvas frameloop={inView ? "always" : "never"} camera={{ position: [0, 0.35, 4.2], fov: 31 }} dpr={[1, 1.5]} gl={{ alpha: true, antialias: true }}
      aria-label="Crew member figure; the lungs move with the breathing signal measured from Wi-Fi">
      <hemisphereLight args={["#ffffff", "#c9d2e0", 0.95]} />
      <directionalLight position={[2.2, 4, 3]} intensity={1.35} color="#fffaf3" />
      <directionalLight position={[-3, 1.2, 2]} intensity={0.45} color="#eef4ff" />
      <directionalLight position={[0, 2.5, -4]} intensity={1.6} color={colors.rim} />
      <group position={[0, -0.22, 0]}>
        <Figure crewId={p.crewId} framesRef={p.framesRef} history={p.history} lastFrameAt={p.lastFrameAt}
          speedRef={p.speedRef} anchors={anchors} colors={colors} />
        <Router x={-0.95} colors={colors} anchorRef={(o) => (anchors.router = o)} />
        <Router x={0.95} colors={colors} />
        <Rays anchors={anchors} colors={colors} />
      </group>
      <Projector anchors={anchors} els={p.calloutEls} />
    </Canvas>
    </div>
  );
}
