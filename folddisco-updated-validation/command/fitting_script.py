# Utility
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
    lam = _newton_with_bisect(lambda L: _lawless416(x, L))
    m = np.mean(np.exp(-lam * x))
    mu = -math.log(m) / lam

    return mu, lam

def evd_sf(x, mu, lam):
    x = np.asarray(x, float)
    y = lam * (x - mu)
    t = np.exp(-y)
    sf = 1 - np.exp(-t)
    sf[y > 20] = np.exp(-y[y > 20])
    return sf

# Worker: compute p-values + plot → return only rows

def per_row_output(file_path):
    mu_L, lam_L = evd_mle_full(scores_L)
    for cols in rows:
        # --- p-value using μ(L), λ(L)
        if np.isfinite(mu_L) and np.isfinite(lam_L) and lam_L > 0:
            p = evd_sf([score], mu_L, lam_L)[0]
            p = float(np.clip(p, 1e-300, 1.0))
        else:
            p = 1.0

        # e-value: M = number of total scores for this length (not key)
        M = len(scores_L)
        e = M * p

    #L_fixed = number of residues
    #rows = list for all rows
    #scores_L = list for all scores

    # -------------------------------
    # STEP 2: Fit μ(L), λ(L) directly from all scores of this length
    # -------------------------------
    

    # -------------------------------
    # STEP 3: Compute p-values for each row
    # -------------------------------

