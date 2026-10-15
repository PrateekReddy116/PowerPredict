import { useCallback, useEffect, useState } from "react";

import { MODEL_LABELS, MODEL_ORDER, isModelName, type ModelName } from "../lib/models";

const KEY = "pp_model";

export function useSelectedModel(): {
  model: ModelName;
  setModel: (name: string) => void;
} {
  const [model, setName] = useState<ModelName>(() => {
    const stored = sessionStorage.getItem(KEY);
    return stored && isModelName(stored) ? stored : "xgboost";
  });

  useEffect(() => {
    sessionStorage.setItem(KEY, model);
  }, [model]);

  const setModel = useCallback((name: string) => {
    if (isModelName(name)) setName(name);
  }, []);

  return { model, setModel };
}

export const MODEL_SELECT_OPTIONS = MODEL_ORDER.map((value) => ({
  value,
  label: MODEL_LABELS[value] ?? value,
}));
