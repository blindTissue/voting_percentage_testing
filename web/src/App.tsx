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
type Language = 'en' | 'ko'

type DragTarget = {
  district: 'a' | 'b'
  clusterId: string
}

const SQRT3_OVER_2 = Math.sqrt(3) / 2
const GRID_SIZE = 34

const COPY = {
  en: {
    eyebrow: 'vote count probability explorer',
    title: 'Same Vote Count Probability Explorer',
    intro:
      'Some Korean election districts reported identical vote counts for two candidates, raising public questions about how unlikely that is. This tool lets you explore that question under Dirichlet-Multinomial probability distribution. Choose a model, adjust the vote-share distributions for two districts, and calculate the chance that both districts produce the same counts for candidate 1 and candidate 2. This does not prove or disprove fraud; it shows how the probability changes as your assumptions change.',
    singlePattern: 'Single population bloc',
    multipleBlocs: 'Multiple voter blocs',
    distributionModel: 'distribution model',
    voterBlocCount: 'voter bloc count',
    twoClusters: '2 clusters',
    threeClusters: '3 clusters',
    computeMode: 'compute mode',
    fastApproximation: 'Fast approximation',
    exactMixtureSum: 'Exact mixture sum',
    concentrationScale: 'concentration scale',
    linearSlider: 'Linear slider',
    logSlider: 'Log slider',
    exactWarning:
      'Exact mode can freeze the browser for large totals, especially with multiple voter blocs. Use it for small totals or when you are willing to wait.',
    singlePlotHelp: 'Drag the center point to change the expected vote shares.',
    mixturePlotHelp: 'Drag numbered blocs to change each group of voters.',
    ternaryAria: (label: string) => `${label} ternary distribution plot`,
    candidate1: 'candidate 1',
    candidate2: 'candidate 2',
    other: 'other',
    district1: 'District 1',
    district2: 'District 2',
    district1Assumptions: 'District 1 assumptions',
    district2Assumptions: 'District 2 assumptions',
    totalVotes: 'Total votes',
    cluster: (index: number) => `Cluster ${index}`,
    blocWeight: 'Bloc weight',
    candidate1Share: 'Candidate 1 share',
    candidate2Share: 'Candidate 2 share',
    concentration: 'Concentration',
    clusterNote: (other: string, alpha: string) => `Other share = ${other}; model alpha = ${alpha}`,
    result: 'result',
    resultTitle: 'Chance of the two districts matching on both candidate counts',
    exactDescription: 'This calculation checks every possible matching count. It may take a while for large districts.',
    approxDescription: 'This calculation uses a fast estimate so you can explore large districts interactively.',
    computing: 'Computing...',
    computeExact: 'Compute exact probability',
    computeApprox: 'Compute approximate probability',
    probability: 'probability',
    scale: 'scale',
  },
  ko: {
    eyebrow: '득표수 확률 탐색 도구',
    title: '동일 득표수 확률 탐색기',
    intro:
      '최근 일부 선거구에서 두 후보의 득표수가 완전히 똑같이 나오는 사례가 발생해, 이것이 과연 통계적으로 얼마나 일어나기 힘든 일인지 궁금해하는 분들이 있습니다. 이 도구는 통계 모델(디리클레-다항 분포)을 바탕으로 그 확률을 직접 탐색해 볼 수 있게 해줍니다. 예상 득표율과 유권자 그룹을 조정해 가며 조건에 따라 일치 확률이 어떻게 변하는지 확인해 보세요. 이 도구는 선거 조작 여부를 증명하는 것이 아니라, 설정한 조건에 따라 결과가 어떻게 달라지는지 직관적으로 보여주기 위해 만들어졌습니다',
    singlePattern: '단일 유권자 집단',
    multipleBlocs: '다중 유권자 집단',
    distributionModel: '분포 모델',
    voterBlocCount: '유권자 집단 수',
    twoClusters: '집단 2개',
    threeClusters: '집단 3개',
    computeMode: '계산 방식',
    fastApproximation: '빠른 근사',
    exactMixtureSum: '정확한 합산',
    concentrationScale: '결집도 슬라이더',
    linearSlider: '선형 슬라이더',
    logSlider: '로그 슬라이더',
    exactWarning:
      '정확한 합산은 총투표수가 크거나 유권자 집단이 여러 개일 때 브라우저가 멈춘 것처럼 보일 수 있습니다. 작은 총투표수에서 사용하거나 시간이 걸릴 수 있음을 감안해 주세요.',
    singlePlotHelp: '중심점을 드래그해 예상 득표율을 바꿀 수 있습니다.',
    mixturePlotHelp: '번호가 붙은 집단을 드래그해 각 유권자 집단을 조정할 수 있습니다.',
    ternaryAria: (label: string) => `${label} 삼각 분포 그래프`,
    candidate1: '후보 1',
    candidate2: '후보 2',
    other: '기타',
    district1: '선거구 1',
    district2: '선거구 2',
    district1Assumptions: '선거구 1 설정',
    district2Assumptions: '선거구 2 설정',
    totalVotes: '총투표수',
    cluster: (index: number) => `집단 ${index}`,
    blocWeight: '집단 비중',
    candidate1Share: '후보 1 득표율',
    candidate2Share: '후보 2 득표율',
    concentration: '결집도',
    clusterNote: (other: string, alpha: string) => `기타 비율 = ${other}; 모델 알파 = ${alpha}`,
    result: '결과',
    resultTitle: '두 선거구가 두 후보 득표수 모두에서 일치할 가능성',
    exactDescription: '이 계산은 일치할 수 있는 모든 득표수 조합을 확인합니다. 큰 선거구에서는 시간이 걸릴 수 있습니다.',
    approxDescription: '이 계산은 빠른 추정값을 사용하므로 큰 선거구도 빠르게 살펴볼 수 있습니다.',
    computing: '계산 중...',
    computeExact: '정밀 확률 계산하기',
    computeApprox: '빠른 확률 추정하기',
    probability: '확률',
    scale: '발생 빈도',
  },
}

type Copy = (typeof COPY)[Language]

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

function formatProbability(value: number | null, language: Language) {
  if (value === null) return ['-', '-']
  const inverse = (1 / value).toLocaleString(language === 'ko' ? 'ko-KR' : undefined, { maximumFractionDigits: 1 })
  return [value.toExponential(6), language === 'ko' ? `${inverse}번 중 1번` : `1 in ${inverse}`]
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
  copy,
  dragTarget,
  onDragStart,
  onDrag,
  onDragEnd,
}: {
  district: District
  label: string
  mode: ModelMode
  copy: Copy
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
        <p>{mode === 'single' ? copy.singlePlotHelp : copy.mixturePlotHelp}</p>
      </div>
      <svg
        ref={svgRef}
        className="ternary"
        viewBox="-0.08 -0.08 1.16 1.02"
        role="img"
        aria-label={copy.ternaryAria(label)}
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
          {copy.candidate1}
        </text>
        <text x="1.02" y={SQRT3_OVER_2 + 0.045} className="axis-label right">
          {copy.candidate2}
        </text>
        <text x="0.5" y="-0.025" className="axis-label top">
          {copy.other}
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
  copy,
  onChange,
}: {
  label: string
  district: District
  clusterCount: number
  mode: ModelMode
  kappaScale: KappaScale
  copy: Copy
  onChange: (district: District) => void
}) {
  const visibleClusters = mode === 'single' ? district.clusters.slice(0, 1) : district.clusters.slice(0, clusterCount)
  return (
    <section className="control-card">
      <div className="control-title">
        <h2>{label}</h2>
        <label>
          {copy.totalVotes}
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
          <div className={`cluster-controls ${mode}`} key={cluster.id}>
            <h3>{copy.cluster(index + 1)}</h3>
            {mode === 'mixture' && (
              <label>
                {copy.blocWeight}
                <input
                  type="number"
                  min="0"
                  step="0.05"
                  value={cluster.weight}
                  onChange={(event) => onChange(updateCluster(district, cluster.id, { weight: Number(event.target.value) }))}
                />
              </label>
            )}
            <label>
              {copy.candidate1Share}
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
              {copy.candidate2Share}
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
              {copy.concentration}
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
              {copy.clusterNote(other.toFixed(3), clusterAlpha(cluster).map((value) => value.toFixed(2)).join(', '))}
            </p>
          </div>
        )
      })}
    </section>
  )
}

function App() {
  const [language, setLanguage] = useState<Language>('en')
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

  const [probabilityText, probabilityScale] = formatProbability(probability, language)
  const copy = COPY[language]

  return (
    <main>
      <div className="language-picker" role="group" aria-label="Language / 언어">
        <button
          type="button"
          className={language === 'en' ? 'selected' : ''}
          aria-label="EN English"
          aria-pressed={language === 'en'}
          onClick={() => setLanguage('en')}
        >
          <span>EN</span>
          English
        </button>
        <button
          type="button"
          className={language === 'ko' ? 'selected' : ''}
          aria-label="KO 한국어"
          aria-pressed={language === 'ko'}
          onClick={() => setLanguage('ko')}
        >
          <span>KO</span>
          한국어
        </button>
      </div>
      <header className="page-header">
        {/* <p className="eyebrow">{copy.eyebrow}</p> */}
        <h1>{copy.title}</h1>
        <p>{copy.intro}</p>
        <div className="mode-toggle" role="group" aria-label={copy.distributionModel}>
          <button type="button" className={modelMode === 'single' ? 'selected' : ''} onClick={() => { setModelMode('single'); setProbability(null) }}>
            {copy.singlePattern}
          </button>
          <button type="button" className={modelMode === 'mixture' ? 'selected' : ''} onClick={() => { setModelMode('mixture'); setProbability(null) }}>
            {copy.multipleBlocs}
          </button>
        </div>
        <div className="header-controls">
          {modelMode === 'mixture' && (
            <label>
              {copy.voterBlocCount}
              <select
                value={clusterCount}
                onChange={(event) => {
                  setClusterCount(Number(event.target.value))
                  setProbability(null)
                }}
              >
                <option value={2}>{copy.twoClusters}</option>
                <option value={3}>{copy.threeClusters}</option>
              </select>
            </label>
          )}
          <label>
            {copy.computeMode}
            <select
              value={computeMode}
              onChange={(event) => {
                setComputeMode(event.target.value as ComputeMode)
                setProbability(null)
              }}
            >
              <option value="approx">{copy.fastApproximation}</option>
              <option value="exact">{copy.exactMixtureSum}</option>
            </select>
          </label>
          <label>
            {copy.concentrationScale}
            <select
              value={kappaScale}
              onChange={(event) => {
                setKappaScale(event.target.value as KappaScale)
                setProbability(null)
              }}
            >
              <option value="linear">{copy.linearSlider}</option>
              <option value="log">{copy.logSlider}</option>
            </select>
          </label>
        </div>
        {computeMode === 'exact' && (
          <p className="warning-note">
            {copy.exactWarning}
          </p>
        )}
      </header>

      <section className="plot-grid">
        <TernaryPlot
          district={activeDistrict(districtA, modelMode, clusterCount)}
          label={copy.district1}
          mode={modelMode}
          copy={copy}
          dragTarget={dragTarget?.district === 'a' ? dragTarget : null}
          onDragStart={(clusterId) => setDragTarget({ district: 'a', clusterId })}
          onDrag={applyDrag}
          onDragEnd={() => setDragTarget(null)}
        />
        <TernaryPlot
          district={activeDistrict(districtB, modelMode, clusterCount)}
          label={copy.district2}
          mode={modelMode}
          copy={copy}
          dragTarget={dragTarget?.district === 'b' ? dragTarget : null}
          onDragStart={(clusterId) => setDragTarget({ district: 'b', clusterId })}
          onDrag={applyDrag}
          onDragEnd={() => setDragTarget(null)}
        />
      </section>

      <section className="control-grid">
        <DistrictControls label={copy.district1Assumptions} district={districtA} clusterCount={clusterCount} mode={modelMode} kappaScale={kappaScale} copy={copy} onChange={setDistrictA} />
        <DistrictControls label={copy.district2Assumptions} district={districtB} clusterCount={clusterCount} mode={modelMode} kappaScale={kappaScale} copy={copy} onChange={setDistrictB} />
      </section>

      <section className="result-panel">
        <div>
          {/* <p className="eyebrow">{copy.result}</p> */}
          <h2>{copy.resultTitle}</h2>
          <p>{computeMode === 'exact' ? copy.exactDescription : copy.approxDescription}</p>
        </div>
        <button type="button" onClick={computeProbability} disabled={isComputing}>
          {isComputing ? copy.computing : computeMode === 'exact' ? copy.computeExact : copy.computeApprox}
        </button>
        <div className="metric">
          <span>{copy.probability}</span>
          <strong>{probabilityText}</strong>
        </div>
        <div className="metric">
          <span>{copy.scale}</span>
          <strong>{probabilityScale}</strong>
        </div>
      </section>
    </main>
  )
}

export default App
