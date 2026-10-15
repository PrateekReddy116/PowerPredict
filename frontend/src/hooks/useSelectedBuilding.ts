import { useCallback, useEffect, useState } from "react";

const KEY = "pp_building_id";

export function useSelectedBuilding(ids: string[]): {
  buildingId: string | null;
  setBuildingId: (id: string) => void;
} {
  const [buildingId, setId] = useState<string | null>(() => sessionStorage.getItem(KEY));

  useEffect(() => {
    if (ids.length === 0) {
      setId(null);
      return;
    }
    if (!buildingId || !ids.includes(buildingId)) {
      const next = ids[0] ?? null;
      setId(next);
      if (next) sessionStorage.setItem(KEY, next);
    }
  }, [ids, buildingId]);

  const setBuildingId = useCallback((id: string) => {
    sessionStorage.setItem(KEY, id);
    setId(id);
  }, []);

  return { buildingId, setBuildingId };
}
