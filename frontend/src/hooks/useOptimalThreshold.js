import { useState, useEffect } from "react";
import { getModelStats } from "../api/client.js";
import { DEFAULT_THRESHOLD } from "../utils/prediction.js";

/**
 * Loads the optimal threshold from /model/stats once on mount.
 * Falls back to DEFAULT_THRESHOLD if the endpoint is unavailable.
 *
 * Returns { threshold, setThreshold, optimalThresh, isOptimal }
 */
export function useOptimalThreshold() {
  const [threshold, setThreshold]       = useState(DEFAULT_THRESHOLD);
  const [optimalThresh, setOptimalThresh] = useState(null);

  useEffect(() => {
    getModelStats()
      .then((stats) => {
        const t = stats.optimal_threshold ?? DEFAULT_THRESHOLD;
        setOptimalThresh(t);
        setThreshold(t);
      })
      .catch(() => {});
  }, []);

  const isOptimal =
    optimalThresh !== null && Math.abs(threshold - optimalThresh) < 0.001;

  return { threshold, setThreshold, optimalThresh, isOptimal };
}
