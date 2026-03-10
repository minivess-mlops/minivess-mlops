import { useEffect, useState } from "react";
import { fetchApi } from "../lib/api";

interface UseAdapterDataResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useAdapterData<T>(endpoint: string): UseAdapterDataResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const result = await fetchApi<T>(endpoint);
        if (!cancelled) {
          setData(result);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();

    // Auto-refresh (read interval from env or default 30s)
    const interval = setInterval(load, 30_000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [endpoint]);

  return { data, loading, error };
}
