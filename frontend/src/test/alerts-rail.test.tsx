import { render, screen } from '@testing-library/react'

import { AlertsRail } from '../components/alerts/AlertsRail'

test('renders alert code and severity', () => {
  render(<AlertsRail alerts={[{ code: 'LOW_SOC', severity: 'critical' }]} />)
  expect(screen.getByText('LOW_SOC')).toBeInTheDocument()
})
