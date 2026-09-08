import { test, expect } from 'vitest';
import { render } from '@testing-library/react';
import App from './App';

test('renders setup controls on the home page', () => {
  const { getByText } = render(<App />);
  expect(getByText(/Map Template/i)).toBeInTheDocument();
  expect(getByText(/Points to Win/i)).toBeInTheDocument();
  expect(getByText(/Card Discard Limit/i)).toBeInTheDocument();
  expect(getByText(/Open hands. Random discard choice./i)).toBeInTheDocument();
  expect(getByText(/^Players$/i)).toBeInTheDocument();
  expect(getByText(/^Start$/i)).toBeInTheDocument();
});
