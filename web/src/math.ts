export type Barycentric = [number, number, number]
export type Point = { x: number; y: number }

const SQRT3_OVER_2 = Math.sqrt(3) / 2
const LOG_SQRT_TWO_PI = 0.9189385332046727

export function alphaFromMean(u1: number, u2: number, kappa: number): Barycentric {
  if (kappa <= 0) {
    throw new Error('kappa must be positive')
  }
  if (u1 <= 0 || u2 <= 0) {
    throw new Error('u1 and u2 must be positive')
  }
  const other = 1 - u1 - u2
  if (other <= 1e-12) {
    throw new Error('u1 + u2 must be less than 1')
  }
  return [kappa * u1, kappa * u2, kappa * other]
}

export function barycentricToSvg([, u2, other]: Barycentric): Point {
  return {
    x: u2 + 0.5 * other,
    y: SQRT3_OVER_2 * other,
  }
}

export function svgToBarycentric(x: number, y: number): Barycentric {
  const other = y / SQRT3_OVER_2
  const u2 = x - 0.5 * other
  const u1 = 1 - u2 - other
  return projectToSimplex([u1, u2, other])
}

export function svgScreenToBarycentric(x: number, y: number): Barycentric {
  return svgToBarycentric(x, SQRT3_OVER_2 - y)
}

export function projectToSimplex(values: Barycentric): Barycentric {
  const clipped = values.map((value) => Math.max(0.001, value)) as Barycentric
  const total = clipped[0] + clipped[1] + clipped[2]
  return [clipped[0] / total, clipped[1] / total, clipped[2] / total]
}

export function logGamma(z: number): number {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
  ]

  if (z < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z)
  }

  let x = 0.9999999999998099
  const shifted = z - 1
  for (let i = 0; i < coefficients.length; i += 1) {
    x += coefficients[i] / (shifted + i + 1)
  }
  const t = shifted + coefficients.length - 0.5
  return LOG_SQRT_TWO_PI + (shifted + 0.5) * Math.log(t) - t + Math.log(x)
}

function logComb(n: number, k: number): number {
  return logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1)
}

function binomLogPmf(k: number, n: number, p: number): number {
  return logComb(n, k) + k * Math.log(p) + (n - k) * Math.log1p(-p)
}

function betaBinomLogPmf(k: number, n: number, a: number, b: number): number {
  return (
    logComb(n, k) +
    logGamma(k + a) +
    logGamma(n - k + b) -
    logGamma(n + a + b) +
    logGamma(a + b) -
    logGamma(a) -
    logGamma(b)
  )
}

function logSumExp(values: number[]): number {
  const max = Math.max(...values)
  if (!Number.isFinite(max)) return max
  let total = 0
  for (const value of values) {
    total += Math.exp(value - max)
  }
  return max + Math.log(total)
}

export function collisionProbabilityBinomial(nA: number, nB: number, pA: number, pB: number): number {
  const maxHeads = Math.min(nA, nB)
  const terms: number[] = []
  for (let heads = 0; heads <= maxHeads; heads += 1) {
    terms.push(binomLogPmf(heads, nA, pA) + binomLogPmf(heads, nB, pB))
  }
  return Math.exp(logSumExp(terms))
}

export function collisionProbabilityDirichletMultinomial(
  nA: number,
  nB: number,
  alphaA: Barycentric,
  alphaB: Barycentric,
): number {
  const maxCandidate1 = Math.min(nA, nB)
  const terms: number[] = []

  for (let candidate1 = 0; candidate1 <= maxCandidate1; candidate1 += 1) {
    const maxCandidate2 = Math.min(nA - candidate1, nB - candidate1)
    const conditionalTerms: number[] = []
    for (let candidate2 = 0; candidate2 <= maxCandidate2; candidate2 += 1) {
      conditionalTerms.push(
        betaBinomLogPmf(candidate2, nA - candidate1, alphaA[1], alphaA[2]) +
          betaBinomLogPmf(candidate2, nB - candidate1, alphaB[1], alphaB[2]),
      )
    }
    terms.push(
      betaBinomLogPmf(candidate1, nA, alphaA[0], alphaA[1] + alphaA[2]) +
        betaBinomLogPmf(candidate1, nB, alphaB[0], alphaB[1] + alphaB[2]) +
        logSumExp(conditionalTerms),
    )
  }

  return Math.exp(logSumExp(terms))
}

export function collisionProbabilityMixture(
  nA: number,
  nB: number,
  weightsA: number[],
  alphasA: Barycentric[],
  weightsB: number[],
  alphasB: Barycentric[],
): number {
  const normalizedA = normalizeWeights(weightsA)
  const normalizedB = normalizeWeights(weightsB)
  let total = 0
  for (let i = 0; i < alphasA.length; i += 1) {
    for (let j = 0; j < alphasB.length; j += 1) {
      total +=
        normalizedA[i] *
        normalizedB[j] *
        collisionProbabilityDirichletMultinomial(nA, nB, alphasA[i], alphasB[j])
    }
  }
  return total
}

function dmMoments(n: number, alpha: Barycentric) {
  const alpha0 = alpha[0] + alpha[1] + alpha[2]
  const p1 = alpha[0] / alpha0
  const p2 = alpha[1] / alpha0
  const scale = (n * (n + alpha0)) / (alpha0 + 1)
  return {
    mean1: n * p1,
    mean2: n * p2,
    var1: scale * p1 * (1 - p1),
    var2: scale * p2 * (1 - p2),
    cov12: -scale * p1 * p2,
  }
}

function bivariateNormalLatticeAtZero(nA: number, nB: number, alphaA: Barycentric, alphaB: Barycentric): number {
  const a = dmMoments(nA, alphaA)
  const b = dmMoments(nB, alphaB)
  const mean1 = a.mean1 - b.mean1
  const mean2 = a.mean2 - b.mean2
  const var1 = a.var1 + b.var1
  const var2 = a.var2 + b.var2
  const cov12 = a.cov12 + b.cov12
  const determinant = var1 * var2 - cov12 * cov12
  if (determinant <= 0) return 0
  const quadratic = (var2 * mean1 * mean1 - 2 * cov12 * mean1 * mean2 + var1 * mean2 * mean2) / determinant
  return Math.exp(-0.5 * quadratic) / (2 * Math.PI * Math.sqrt(determinant))
}

export function collisionProbabilityApprox(
  nA: number,
  nB: number,
  weightsA: number[],
  alphasA: Barycentric[],
  weightsB: number[],
  alphasB: Barycentric[],
): number {
  const normalizedA = normalizeWeights(weightsA)
  const normalizedB = normalizeWeights(weightsB)
  let total = 0
  for (let i = 0; i < alphasA.length; i += 1) {
    for (let j = 0; j < alphasB.length; j += 1) {
      total += normalizedA[i] * normalizedB[j] * bivariateNormalLatticeAtZero(nA, nB, alphasA[i], alphasB[j])
    }
  }
  return total
}

export function normalizeWeights(weights: number[]): number[] {
  const total = weights.reduce((sum, weight) => sum + Math.max(0, weight), 0)
  if (total <= 0) return weights.map(() => 1 / weights.length)
  return weights.map((weight) => Math.max(0, weight) / total)
}
