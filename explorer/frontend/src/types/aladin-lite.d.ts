declare module 'aladin-lite' {
  const A: {
    init: Promise<void>;
    aladin: (container: HTMLElement, options: Record<string, unknown>) => unknown;
    catalog: (options: Record<string, unknown>) => unknown;
    source: (ra: number, dec: number, data: Record<string, unknown>) => unknown;
  };
  export default A;
}
