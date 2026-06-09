/**
 * @vitest-environment jsdom
 */
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import { afterEach, describe, expect, it } from 'vitest'
import App from './App'

afterEach(() => {
  cleanup()
})

describe('App language selector', () => {
  it('keeps both language choices readable while switching the app copy', () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: /same vote count probability explorer/i })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /KO 한국어/ }))

    expect(screen.getByRole('heading', { name: '동일 득표수 확률 탐색기' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /EN English/ })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /EN English/ }))

    expect(screen.getByRole('heading', { name: /same vote count probability explorer/i })).toBeInTheDocument()
  })
})

describe('single-population controls', () => {
  it('does not show bloc weight inputs until the multiple-bloc model is selected', () => {
    const { container } = render(<App />)

    expect(screen.queryAllByLabelText(/bloc weight/i)).toHaveLength(0)
    expect(container.querySelectorAll('.cluster-controls.single')).toHaveLength(2)

    fireEvent.click(screen.getByRole('button', { name: /multiple voter blocs/i }))

    expect(screen.getAllByLabelText(/bloc weight/i)).toHaveLength(4)
    expect(container.querySelectorAll('.cluster-controls.mixture')).toHaveLength(4)
  })
})
