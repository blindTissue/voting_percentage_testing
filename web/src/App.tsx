import { useMemo, useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import './App.css'
import {
  alphaFromMean,
  barycentricToSvg,
  collisionProbabilityApprox,
  collisionProbabilityMixture,
  logGamma,
  normalizeWeights,
  svgToBarycentric,
} from './math'
import type { Barycentric } from './math'

type Cluster = {
  id: string
  weight: number
  u1: number
  u2: number
  kappa: number
}

type District = {
  totalVotes: number
  clusters: Cluster[]
}

type ModelMode = 'single' | 'mixture'
type ComputeMode = 'approx' | 'exact'
type KappaScale = 'linear' | 'log'

type DragTarget = {
  district: 'a' | 'b'
  clusterId: string
}

const SQRT3_OVER_2 = Math.sqrt(3) / 2
const GRID_SIZE = 34

const initialA: District = {
  totalVotes: 4548,
  clusters: [
    { id: 'a-1', weight: 0.5, u1: 0.666, u2: 0.317, kappa: 80 },
    { id: 'a-2', weight: 0.25, u1: 0.55, u2: 0.418, kappa: 60 },
    { id: 'a-3', weight: 0.25, u1: 0.77, u2: 0.187, kappa: 60 },
  ],
}

const initialB: District = {
  totalVotes: 4540,
  clusters: [
    { id: 'b-1', weight: 0.5, u1: 0.667, u2: 0.317, kappa: 80 },
    { id: 'b-2', weight: 0.25, u1: 0.548, u2: 0.419, kappa: 60 },
    { id: 'b-3', weight: 0.25, u1: 0.769, u2: 0.187, kappa: 60 },
  ],
}

function clusterOther(cluster: Cluster) {
  return 1 - cluster.u1 - cluster.u2
}

function clusterAlpha(cluster: Cluster): Barycentric {
  return alphaFromMean(cluster.u1, cluster.u2, cluster.kappa)
}

function formatProbability(value: number | null) {
  if (value === null) return ['-', '-']
  return [value.toExponential(6), `1 in ${(1 / value).toLocaleString(undefined, { maximumFractionDigits: 1 })}`]
}

function simplexGrid() {
  const points: Barycentric[] = []
  for (let i = 1; i < GRID_SIZE; i += 1) {
    for (let j = 1; j < GRID_SIZE - i; j += 1) {
      const k = GRID_SIZE - i - j
      if (k > 0) points.push([i / GRID_SIZE, j / GRID_SIZE, k / GRID_SIZE])
    }
  }
  return points
}

function dirichletLogDensity(point: Barycentric, alpha: Barycentric) {
  const logNorm = logGamma(alpha[0] + alpha[1] + alpha[2]) - alpha.reduce((sum, value) => sum + logGamma(value), 0)
  return logNorm + (alpha[0] - 1) * Math.log(point[0]) + (alpha[1] - 1) * Math.log(point[1]) + (alpha[2] - 1) * Math.log(point[2])
}

function heatPoints(district: District) {
  const grid = simplexGrid()
  const weights = normalizeWeights(district.clusters.map((cluster) => cluster.weight))
  const componentHeat = district.clusters.map((cluster, index) => {
    const alpha = clusterAlpha(cluster)
    const logs = grid.map((point) => dirichletLogDensity(point, alpha))
    const maxLog = Math.max(...logs)
    return logs.map((value) => weights[index] * Math.exp(value - maxLog))
  })
  const values = grid.map((_, pointIndex) => componentHeat.reduce((sum, values) => sum + values[pointIndex], 0))
  const maxValue = Math.max(...values)
  return grid.map((point, index) => ({ point, value: values[index] / maxValue }))
}

function activeDistrict(district: District, mode: ModelMode, clusterCount: number): District {
  return mode === 'single' ? { ...district, clusters: [district.clusters[0]] } : { ...district, clusters: district.clusters.slice(0, clusterCount) }
}

function heatColor(value: number) {
  const hue = 44 - 34 * value
  const lightness = 94 - 43 * value
  return `hsl(${hue} 92% ${lightness}%)`
}

function updateCluster(district: District, clusterId: string, patch: Partial<Cluster>): District {
  return {
    ...district,
    clusters: district.clusters.map((cluster) => (cluster.id === clusterId ? { ...cluster, ...patch } : cluster)),
  }
}

function kappaToSlider(kappa: number, scale: KappaScale) {
  return scale === 'log' ? Math.log10(kappa) : kappa
}

function sliderToKappa(value: number, scale: KappaScale) {
  return scale === 'log' ? 10 ** value : value
}

function TernaryPlot({
  district,
  label,
  mode,
  dragTarget,
  onDragStart,
  onDrag,
  onDragEnd,
}: {
  district: District
  label: string
  mode: ModelMode
  dragTarget: DragTarget | null
  onDragStart: (clusterId: string) => void
  onDrag: (u: Barycentric) => void
  onDragEnd: () => void
}) {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const heat = useMemo(() => heatPoints(district), [district])
  const normalizedWeights = normalizeWeights(district.clusters.map((cluster) => cluster.weight))

  const handlePointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (!dragTarget || !svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const x = (event.clientX - rect.left) / rect.width
    const screenY = ((event.clientY - rect.top) / rect.height) * SQRT3_OVER_2
    const modelY = SQRT3_OVER_2 - screenY
    onDrag(svgToBarycentric(x, modelY))
  }

  return (
    <section className="plot-card">
      <div className="plot-heading">
        <h2>{label}</h2>
        <p>{mode === 'single' ? 'Drag the center; u values update live.' : 'Drag numbered cluster centers; u values update live.'}</p>
      </div>
      <svg
        ref={svgRef}
        className="ternary"
        viewBox="-0.08 -0.08 1.16 1.02"
        role="img"
        aria-label={`${label} ternary distribution plot`}
        onPointerMove={handlePointerMove}
        onPointerUp={onDragEnd}
        onPointerLeave={onDragEnd}
      >
        <polygon points={`0,${SQRT3_OVER_2} 1,${SQRT3_OVER_2} 0.5,0`} className="triangle-bg" />
        {heat.map(({ point, value }) => {
          const position = barycentricToSvg(point)
          return (
            <circle
              key={`${point[0]}-${point[1]}-${point[2]}`}
              cx={position.x}
              cy={SQRT3_OVER_2 - position.y}
              r="0.018"
              fill={heatColor(value)}
            />
          )
        })}
        <polygon points={`0,${SQRT3_OVER_2} 1,${SQRT3_OVER_2} 0.5,0`} className="triangle-line" />
        <text x="-0.02" y={SQRT3_OVER_2 + 0.045} className="axis-label">
          candidate 1
        </text>
        <text x="1.02" y={SQRT3_OVER_2 + 0.045} className="axis-label right">
          candidate 2
        </text>
        <text x="0.5" y="-0.025" className="axis-label top">
          other
        </text>
        {district.clusters.map((cluster, index) => {
          const position = barycentricToSvg([cluster.u1, cluster.u2, clusterOther(cluster)])
          const active = dragTarget?.clusterId === cluster.id
          return (
            <g
              key={cluster.id}
              className={`cluster-handle ${active ? 'active' : ''}`}
              transform={`translate(${position.x} ${SQRT3_OVER_2 - position.y})`}
              onPointerDown={(event) => {
                event.currentTarget.setPointerCapture(event.pointerId)
                onDragStart(cluster.id)
              }}
            >
              <circle r={0.035 + normalizedWeights[index] * 0.035} />
              <text y="0.012">{index + 1}</text>
            </g>
          )
        })}
      </svg>
    </section>
  )
}

function DistrictControls({
  label,
  district,
  clusterCount,
  mode,
  kappaScale,
  onChange,
}: {
  label: string
  district: District
  clusterCount: number
  mode: ModelMode
  kappaScale: KappaScale
  onChange: (district: District) => void
}) {
  const visibleClusters = mode === 'single' ? district.clusters.slice(0, 1) : district.clusters.slice(0, clusterCount)
  return (
    <section className="control-card">
      <div className="control-title">
        <h2>{label}</h2>
        <label>
          Total votes
          <input
            type="number"
            min="1"
            value={district.totalVotes}
            onChange={(event) => onChange({ ...district, totalVotes: Number(event.target.value) })}
          />
        </label>
      </div>
      {visibleClusters.map((cluster, index) => {
        const other = clusterOther(cluster)
        return (
          <div className="cluster-controls" key={cluster.id}>
            <h3>Cluster {index + 1}</h3>
            <label>
              weight
              <input
                type="number"
                min="0"
                step="0.05"
                value={cluster.weight}
                onChange={(event) => onChange(updateCluster(district, cluster.id, { weight: Number(event.target.value) }))}
              />
            </label>
            <label>
              u1
              <input
                type="number"
                min="0.001"
                max="0.999"
                step="0.001"
                value={cluster.u1}
                onChange={(event) => onChange(updateCluster(district, cluster.id, { u1: Number(event.target.value) }))}
              />
            </label>
            <label>
              u2
              <input
                type="number"
                min="0.001"
                max="0.999"
                step="0.001"
                value={cluster.u2}
                onChange={(event) => onChange(updateCluster(district, cluster.id, { u2: Number(event.target.value) }))}
              />
            </label>
            <label>
              kappa
              <input
                type="range"
                min={kappaScale === 'log' ? -1 : 0.1}
                max={kappaScale === 'log' ? 5 : 1000}
                step={kappaScale === 'log' ? 0.01 : 0.1}
                value={kappaToSlider(cluster.kappa, kappaScale)}
                onChange={(event) =>
                  onChange(updateCluster(district, cluster.id, { kappa: sliderToKappa(Number(event.target.value), kappaScale) }))
                }
              />
              <span className="slider-value">{cluster.kappa < 100 ? cluster.kappa.toFixed(2) : cluster.kappa.toFixed(0)}</span>
            </label>
            <p className={other > 0 ? 'cluster-note' : 'cluster-note error'}>
              other = {other.toFixed(3)}; alpha = {clusterAlpha(cluster).map((value) => value.toFixed(2)).join(', ')}
            </p>
          </div>
        )
      })}
    </section>
  )
}

function App() {
  const [districtA, setDistrictA] = useState(initialA)
  const [districtB, setDistrictB] = useState(initialB)
  const [modelMode, setModelMode] = useState<ModelMode>('single')
  const [clusterCount, setClusterCount] = useState(2)
  const [computeMode, setComputeMode] = useState<ComputeMode>('approx')
  const [kappaScale, setKappaScale] = useState<KappaScale>('linear')
  const [dragTarget, setDragTarget] = useState<DragTarget | null>(null)
  const [probability, setProbability] = useState<number | null>(null)
  const [isComputing, setIsComputing] = useState(false)

  const applyDrag = (u: Barycentric) => {
    if (!dragTarget) return
    const patch = { u1: Number(u[0].toFixed(4)), u2: Number(u[1].toFixed(4)) }
    if (dragTarget.district === 'a') {
      setDistrictA((district) => updateCluster(district, dragTarget.clusterId, patch))
    } else {
      setDistrictB((district) => updateCluster(district, dragTarget.clusterId, patch))
    }
    setProbability(null)
  }

  const computeProbability = () => {
    setIsComputing(true)
    window.setTimeout(() => {
      const activeA = activeDistrict(districtA, modelMode, clusterCount)
      const activeB = activeDistrict(districtB, modelMode, clusterCount)
      const probabilityFunction = computeMode === 'exact' ? collisionProbabilityMixture : collisionProbabilityApprox
      const value = probabilityFunction(
        activeA.totalVotes,
        activeB.totalVotes,
        activeA.clusters.map((cluster) => cluster.weight),
        activeA.clusters.map(clusterAlpha),
        activeB.clusters.map((cluster) => cluster.weight),
        activeB.clusters.map(clusterAlpha),
      )
      setProbability(value)
      setIsComputing(false)
    }, 20)
  }

  const [probabilityText, probabilityScale] = formatProbability(probability)

  return (
    <main>
      <header className="page-header">
        {/* <p className="eyebrow">Probability that vote </p> */}
        <h1>Same vote count probability calculation</h1>
        <p>
          Recently, there has been some claims of voter fraud in the 9th Korean local elections.
          The claim stems from the same vote count for two candidates across different districts.
          The "probability" that same count comes out differs by claim.
          This is my approach of calculating the probability.
          Like always, statistical models are not real. They are a model of the real world.
          This is a Dirichlet prior, categorical posterior model.
          Drag, change the prior distribution as you feel fit.
        </p>
        <div className="mode-toggle" role="group" aria-label="distribution model">
          <button type="button" className={modelMode === 'single' ? 'selected' : ''} onClick={() => { setModelMode('single'); setProbability(null) }}>
            Single Dirichlet
          </button>
          <button type="button" className={modelMode === 'mixture' ? 'selected' : ''} onClick={() => { setModelMode('mixture'); setProbability(null) }}>
            Mixture clusters
          </button>
        </div>
        <div className="header-controls">
          {modelMode === 'mixture' && (
            <label>
              cluster count
              <select
                value={clusterCount}
                onChange={(event) => {
                  setClusterCount(Number(event.target.value))
                  setProbability(null)
                }}
              >
                <option value={2}>2 clusters</option>
                <option value={3}>3 clusters</option>
              </select>
            </label>
          )}
          <label>
            compute mode
            <select
              value={computeMode}
              onChange={(event) => {
                setComputeMode(event.target.value as ComputeMode)
                setProbability(null)
              }}
            >
              <option value="approx">Fast approximation</option>
              <option value="exact">Exact mixture sum</option>
            </select>
          </label>
          <label>
            kappa scale
            <select
              value={kappaScale}
              onChange={(event) => {
                setKappaScale(event.target.value as KappaScale)
                setProbability(null)
              }}
            >
              <option value="linear">Linear slider</option>
              <option value="log">Log slider</option>
            </select>
          </label>
        </div>
        {computeMode === 'exact' && (
          <p className="warning-note">
            Exact mode can freeze the browser for large totals, especially with mixture clusters. Use it for small totals or when you are willing to wait.
          </p>
        )}
      </header>

      <section className="plot-grid">
        <TernaryPlot
          district={activeDistrict(districtA, modelMode, clusterCount)}
          label="District 1"
          mode={modelMode}
          dragTarget={dragTarget?.district === 'a' ? dragTarget : null}
          onDragStart={(clusterId) => setDragTarget({ district: 'a', clusterId })}
          onDrag={applyDrag}
          onDragEnd={() => setDragTarget(null)}
        />
        <TernaryPlot
          district={activeDistrict(districtB, modelMode, clusterCount)}
          label="District 2"
          mode={modelMode}
          dragTarget={dragTarget?.district === 'b' ? dragTarget : null}
          onDragStart={(clusterId) => setDragTarget({ district: 'b', clusterId })}
          onDrag={applyDrag}
          onDragEnd={() => setDragTarget(null)}
        />
      </section>

      <section className="control-grid">
        <DistrictControls label="District 1 controls" district={districtA} clusterCount={clusterCount} mode={modelMode} kappaScale={kappaScale} onChange={setDistrictA} />
        <DistrictControls label="District 2 controls" district={districtB} clusterCount={clusterCount} mode={modelMode} kappaScale={kappaScale} onChange={setDistrictB} />
      </section>

      <section className="result-panel">
        <div>
          <p className="eyebrow">event</p>
          <h2>Same candidate 1 count and same candidate 2 count</h2>
          <p>{computeMode === 'exact' ? 'Exact mode sums the full Dirichlet-Multinomial mixture and can be slow for large totals.' : 'Fast mode uses a bivariate-normal lattice approximation so large vote totals stay interactive.'}</p>
        </div>
        <button type="button" onClick={computeProbability} disabled={isComputing}>
          {isComputing ? 'Computing...' : computeMode === 'exact' ? 'Compute exact probability' : 'Compute approximate probability'}
        </button>
        <div className="metric">
          <span>probability</span>
          <strong>{probabilityText}</strong>
        </div>
        <div className="metric">
          <span>scale</span>
          <strong>{probabilityScale}</strong>
        </div>
      </section>
    </main>
  )
}

export default App
