import { describe, expect, it } from 'vitest'
import {
  alphaFromMean,
  barycentricToSvg,
  collisionProbabilityApprox,
  collisionProbabilityBinomial,
  collisionProbabilityDirichletMultinomial,
  collisionProbabilityMixture,
  svgToBarycentric,
} from './math'

describe('alphaFromMean', () => {
  it('converts u1, u2, and implied other share into alpha', () => {
    expect(alphaFromMean(0.6, 0.3, 50)[0]).toBeCloseTo(30)
    expect(alphaFromMean(0.6, 0.3, 50)[1]).toBeCloseTo(15)
    expect(alphaFromMean(0.6, 0.3, 50)[2]).toBeCloseTo(5)
  })

  it('rejects invalid mean shares', () => {
    expect(() => alphaFromMean(0.7, 0.3, 50)).toThrow(/less than 1/)
  })
})

describe('ternary coordinate conversion', () => {
  it('round-trips barycentric coordinates through svg coordinates', () => {
    const original: [number, number, number] = [0.62, 0.31, 0.07]
    const point = barycentricToSvg(original)
    const roundTrip = svgToBarycentric(point.x, point.y)

    expect(roundTrip[0]).toBeCloseTo(original[0], 10)
    expect(roundTrip[1]).toBeCloseTo(original[1], 10)
    expect(roundTrip[2]).toBeCloseTo(original[2], 10)
  })
})

describe('collisionProbabilityBinomial', () => {
  it('matches the one-flip fair coin sanity check', () => {
    expect(collisionProbabilityBinomial(1, 1, 0.5, 0.5)).toBeCloseTo(0.5)
  })

  it('matches the two-flip fair coin sanity check', () => {
    expect(collisionProbabilityBinomial(2, 2, 0.5, 0.5)).toBeCloseTo(3 / 8)
  })
})

describe('collisionProbabilityApprox', () => {
  it('is close to the exact Dirichlet-Multinomial result at moderate totals', () => {
    const alphaA: [number, number, number] = [60, 30, 10]
    const alphaB: [number, number, number] = [58, 32, 10]
    const exact = collisionProbabilityDirichletMultinomial(120, 118, alphaA, alphaB)
    const approximate = collisionProbabilityApprox(120, 118, [1], [alphaA], [1], [alphaB])

    expect(approximate / exact).toBeGreaterThan(0.7)
    expect(approximate / exact).toBeLessThan(1.3)
  })
})

describe('collisionProbabilityMixture', () => {
  it('one-component exact mixture equals exact Dirichlet-Multinomial', () => {
    const alphaA: [number, number, number] = [8, 5, 2]
    const alphaB: [number, number, number] = [7, 4, 3]

    expect(collisionProbabilityMixture(7, 6, [1], [alphaA], [1], [alphaB])).toBeCloseTo(
      collisionProbabilityDirichletMultinomial(7, 6, alphaA, alphaB),
      12,
    )
  })
})
