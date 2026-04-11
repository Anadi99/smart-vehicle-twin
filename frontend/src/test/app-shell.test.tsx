import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { AppShell } from '../components/layout/AppShell'

test('renders primary navigation labels', () => {
  render(
    <MemoryRouter>
      <AppShell>
        <div>content</div>
      </AppShell>
    </MemoryRouter>,
  )
  expect(screen.getByText('Overview')).toBeInTheDocument()
  expect(screen.getByText('Telemetry')).toBeInTheDocument()
})
