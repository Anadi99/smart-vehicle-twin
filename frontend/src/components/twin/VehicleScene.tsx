import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei'
import * as THREE from 'three'

import { getHealthColor, healthHueToHex } from './healthColors'

type Props = { latest: Record<string, unknown> }

function num(v: unknown, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback
}

/** Animated car body — orbits the track oval, tilts on acceleration */
function VehicleMesh({ color, speed, accel }: { color: string; speed: number; accel: number }) {
  const groupRef = useRef<THREE.Group>(null!)
  const angle = useRef(0)

  useFrame((_state, delta) => {
    // orbit speed proportional to vehicle speed (1 full lap ≈ 20 s at 180 km/h)
    const angularSpeed = (speed / 180) * (Math.PI * 2) / 20
    angle.current += angularSpeed * delta
    const r = 3.5
    groupRef.current.position.x = Math.cos(angle.current) * r
    groupRef.current.position.z = Math.sin(angle.current) * r
    // face direction of travel
    groupRef.current.rotation.y = -angle.current + Math.PI / 2
    // pitch on acceleration
    groupRef.current.rotation.x = THREE.MathUtils.lerp(
      groupRef.current.rotation.x,
      Math.max(-0.15, Math.min(0.15, accel * 0.03)),
      0.1,
    )
  })

  return (
    <group ref={groupRef}>
      {/* Main body */}
      <mesh position={[0, 0.18, 0]} castShadow>
        <boxGeometry args={[2.0, 0.35, 0.9]} />
        <meshStandardMaterial color={color} metalness={0.4} roughness={0.35} />
      </mesh>
      {/* Cabin */}
      <mesh position={[0.1, 0.45, 0]} castShadow>
        <boxGeometry args={[0.9, 0.28, 0.75]} />
        <meshStandardMaterial color="#1e293b" metalness={0.2} roughness={0.6} />
      </mesh>
      {/* Wheels */}
      {([ [-0.75, -0.05, 0.52], [0.75, -0.05, 0.52], [-0.75, -0.05, -0.52], [0.75, -0.05, -0.52] ] as [number,number,number][]).map(
        ([x, y, z], i) => (
          <mesh key={i} position={[x, y, z]} rotation={[Math.PI / 2, 0, 0]} castShadow>
            <cylinderGeometry args={[0.18, 0.18, 0.14, 16]} />
            <meshStandardMaterial color="#0f172a" roughness={0.9} />
          </mesh>
        ),
      )}
    </group>
  )
}

/** Track oval on the ground */
function TrackOval() {
  const points: THREE.Vector3[] = []
  for (let i = 0; i <= 64; i++) {
    const a = (i / 64) * Math.PI * 2
    points.push(new THREE.Vector3(Math.cos(a) * 3.5, 0.01, Math.sin(a) * 3.5))
  }
  const geo = new THREE.BufferGeometry().setFromPoints(points)
  const mat = new THREE.LineBasicMaterial({ color: '#334155' })
  const lineObj = new THREE.Line(geo, mat)
  return <primitive object={lineObj} />
}

export function VehicleScene({ latest }: Props) {
  const brake = num(latest.brake_temp_c, 35)
  const soc   = num(latest.battery_soc, 82)
  const speed = num(latest.speed_kph, 64)
  const accel = num(latest.accel_mps2, 0)
  const risk  = num(latest.risk, 0)

  const color = healthHueToHex(getHealthColor({ brake_temp_c: brake, battery_soc: soc }))

  return (
    <div className="h-[440px] w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[0, 6, 10]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[8, 10, 6]} intensity={1.2} castShadow />
        <pointLight position={[-4, 3, -4]} intensity={0.4} color="#22d3ee" />

        {/* Ground */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[20, 20]} />
          <meshStandardMaterial color="#0f172a" />
        </mesh>

        <TrackOval />
        <gridHelper args={[20, 20, '#1e293b', '#0f172a']} />

        <VehicleMesh color={color} speed={speed} accel={accel} />

        {/* Risk label */}
        <Text
          position={[0, 3.5, 0]}
          fontSize={0.35}
          color={risk >= 0.7 ? '#f87171' : risk >= 0.4 ? '#fbbf24' : '#34d399'}
          anchorX="center"
          anchorY="middle"
        >
          {`Risk: ${risk.toFixed(2)}  |  ${speed.toFixed(0)} km/h`}
        </Text>

        <OrbitControls enableDamping minDistance={4} maxDistance={20} />
      </Canvas>
    </div>
  )
}
