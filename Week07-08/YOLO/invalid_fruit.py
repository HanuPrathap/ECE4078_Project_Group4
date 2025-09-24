def load_target_set(path="search_list.txt"):
    """
    Return a lowercase set of target fruit/veg names from search_list.txt.
    Blank lines are ignored.
    """
    with open(path, "r") as f:
        return {ln.strip().lower() for ln in f if ln.strip()}


def update_invalid_fruit(detections, target_set, invalid_fruit, min_conf=0.6):
    """
    Add any detected labels NOT in target_set to the invalid_fruit list (once each).

    Parameters
    ----------
    detections : iterable
        Per-frame detector outputs. Each item can be:
          - ("label", conf) tuple/list
          - {"label": str, "conf": float} dict
          - {"name"/"class": str, "confidence"/"score": float} dict
          - plain string label (conf assumed 1.0)
    target_set : set[str]
        Lowercased set of valid target names (e.g., from load_target_set()).
    invalid_fruit : list[str]
        Mutable list you maintain across frames (lowercase names).
    min_conf : float
        Ignore detections below this confidence.

    Returns
    -------
    invalid_fruit : list[str]
        Updated list (same object you passed in).
    newly_added : list[str]
        Labels that were added this call.
    """

    def _extract_label_conf(det):
        # tuple/list: (label, conf?) or (label,)
        if isinstance(det, (list, tuple)):
            if len(det) >= 2:
                return str(det[0]), float(det[1])
            elif len(det) == 1:
                return str(det[0]), 1.0
        # dict with flexible keys
        if isinstance(det, dict):
            label = det.get("label") or det.get("name") or det.get("class")
            conf = (
                det.get("conf", None)
                if det.get("conf", None) is not None
                else det.get("confidence", None)
                if det.get("confidence", None) is not None
                else det.get("score", 1.0)
            )
            return ("" if label is None else str(label)), float(conf)
        # plain string
        return str(det), 1.0

    have = set(s.lower() for s in invalid_fruit)
    newly = []

    for det in detections:
        label, conf = _extract_label_conf(det)
        name = label.strip().lower()
        if not name:
            continue
        if conf < min_conf:
            continue
        if name in target_set:
            continue
        if name not in have:
            invalid_fruit.append(name)
            have.add(name)
            newly.append(name)

    return invalid_fruit, newly
