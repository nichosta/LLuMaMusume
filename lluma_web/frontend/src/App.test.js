import { render, screen } from '@testing-library/react';
import App from './App';

test('renders page header', () => {
  render(<App />);
  const headerElement = screen.getByText(/LLuMa Musume - Agent Monitor/i);
  expect(headerElement).toBeInTheDocument();
});
