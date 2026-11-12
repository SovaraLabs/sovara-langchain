const warned = new Set<string>();

/** Warn once per process for a given key; messages carry metadata only. */
export function warnOnce(key: string, message: string): void {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(message);
}
