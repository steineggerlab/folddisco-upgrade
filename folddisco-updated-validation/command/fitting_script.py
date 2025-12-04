# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def smart_split(line):
    return line.rstrip("\n").split("\t")


def _lawless416(x, lam):
    ex = np.exp(-lam * x)
    esum = ex.sum()
    xesum = (x * ex).sum()
    xxesum = (x * x * ex).sum()
    f  = 1 / lam - x.mean() + xesum / esum
    df = (xesum/esum)**2 - xxesum/esum - 1/(lam**2)
    return f, df


def _newton_with_bisect(f_df, lam0=0.2, tol=1e-5):
    lam = max(lam0, 1e-6)
    for _ in range(80):
        f, df = f_df(lam)
        if abs(f) < tol:
            return lam
        lam = max(lam - f/df, 1e-6)

    L, R = 1e-6, 100
    for _ in range(150):
        M = (L + R) / 2
        fM, _ = f_df(M)
        if abs(fM) < tol:
            return M
        if fM > 0:
            L = M
        else:
            R = M
    return (L + R) / 2


def evd_mle_full(scores):
    x = np.asarray(scores, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        print("small x size")
        return np.nan, np.nan

    std = np.std(x)
    uniq = np.unique(x)

    if np.allclose(x, x[0]) or std < 1e-4:
        return np.nan, np.nan


    lam = _newton_with_bisect(lambda L: _lawless416(x, L))
    if not np.isfinite(lam) or lam <= 0:
        print("lambda weird")
        return np.nan, np.nan

    m = np.mean(np.exp(-lam * x))
    if m <= 0:
        print("mean minus")
        return np.nan, np.nan

    mu = -math.log(m) / lam
    if not np.isfinite(mu):
        print("mu infinite")
        return np.nan, np.nan

    return mu, lam


def evd_sf(x, mu, lam):
    x = np.asarray(x, float)
    y = lam * (x - mu)
    t = np.exp(-y)
    sf = 1 - np.exp(-t)
    sf[y > 20] = np.exp(-y[y > 20])
    return sf


def parse_length_from_filename(path):
    fname = os.path.basename(path)
    m = re.search(r"length_(\d+)_combined", fname)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse length from filename: {fname}")

# ---------------------------------------------------------
# Distinguishing self / similar / non-similar hits
# ---------------------------------------------------------
def subgroup_detector(line):
    cols = smart_split(line)
    if len(cols) < 2:
        return None

    pdb_id = cols[0]
    subgroup = cols[1] + cols[2] + cols[3]

    SUBGROUP_DICT[pdb_id] = subgroup
    return None

# ---------------------------------------------------------
# Worker: compute p-values + plot → return only rows
# ---------------------------------------------------------

def per_row_output(file_path):

    L_fixed = parse_length_from_filename(file_path)
    print(f"[INFO] Processing {file_path} (L={L_fixed})")

    rows = []
    scores_L = []   # <<==== 길이 단일 score bucket

    # -------------------------------
    # STEP 1: Load rows
    # -------------------------------
    with open(file_path) as f:
        header = f.readline()
        for line in f:
            cols = smart_split(line)
            if len(cols) < 14:
                continue

            key = cols[0]
            target = cols[1]

            # extract metrics
            idf = float(cols[3])
            score = idf

            scores_L.append(score)
            rows.append(cols)

    # -------------------------------
    # STEP 2: Fit μ(L), λ(L) directly from all scores of this length
    # -------------------------------
    mu_L, lam_L = evd_mle_full(scores_L)

    # -------------------------------
    # STEP 3: Compute p-values for each row
    # -------------------------------
    out_rows = []
    p_self = []
    p_similar = []
    p_non  = []

    for cols in rows:
        key = cols[0]
        target = cols[1]

        idf = float(cols[3])
        score = idf

        # --- p-value using μ(L), λ(L)
        if np.isfinite(mu_L) and np.isfinite(lam_L) and lam_L > 0:
            p = evd_sf([score], mu_L, lam_L)[0]
            p = float(np.clip(p, 1e-300, 1.0))
        else:
            p = 1.0

        # e-value: M = number of total scores for this length (not key)
        M = len(scores_L)
        e = M * p

        out_rows.append(
            f"{file_path}\t{key}\t{target}\t{L_fixed}\t{score:.6f}\t{e:.3e}\n"
        )

    return out_rows