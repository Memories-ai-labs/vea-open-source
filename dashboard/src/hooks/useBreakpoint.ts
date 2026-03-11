import { useEffect, useState } from 'react';

export function useBreakpoint(maxWidth: number): boolean {
  const getMatch = () =>
    typeof window !== 'undefined' ? window.innerWidth <= maxWidth : false;

  const [matches, setMatches] = useState(getMatch);

  useEffect(() => {
    const onResize = () => setMatches(getMatch());
    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [maxWidth]);

  return matches;
}
