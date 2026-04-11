import { getHealthColor } from '../components/twin/healthColors'

test('returns red for high brake temperature', () => {
  expect(getHealthColor({ brake_temp_c: 110 })).toBe('red')
})
