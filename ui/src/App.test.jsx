import { test, expect } from 'vitest';
import { render } from '@testing-library/react';
import App from './App';

test('renders learn react link', () => {
  const { getByText } = render(<App />);
  const linkElement = getByText(/Play against catanatron/i);
  expect(linkElement).toBeInTheDocument();
});
