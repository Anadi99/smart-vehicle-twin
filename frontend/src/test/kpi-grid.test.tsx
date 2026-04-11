import { render, screen } from '@testing-library/react'

import { KpiGrid } from '../components/kpi/KpiGrid'

test('shows speed value in KPI card', () => {
  render(<KpiGrid latest={{ speed_kph: 121.5 }} />)
  expect(screen.getByText(/121\.5/)).toBeInTheDocument()
})
