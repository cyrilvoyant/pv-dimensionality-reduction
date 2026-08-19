function R = compute_references(D, P)
% Modeles de reference, communs aux deux analyses (filter et wrapper). On les
% calcule UNE fois par horizon et on les insere a l'identique dans les deux
% tables, de sorte que filter et wrapper reposent sur exactement les memes
% baselines.
%
% Cinq references :
%   Persistence_P        y(t+h) = y(t)                      (0 parametre)
%   Persistence_Pcyclic  y(t+h) = y(t+h-T)  (T = 1 jour)    (0 parametre)
%   BLEND_tilde          melange P / P_cyclic pondere par phase (Voyant 2026)
%   ELM_full             ELM sur les 48 retards + temporel   (reference non lineaire)
%   AR_full              moindres carres sur 48 retards + temporel (ref. lineaire)
%
% Colonne FilterScore laissee a NaN (les references ne sont pas geometriques).
    LB     = D.LB;   FH = D.FH;   n_temp = D.n_temp;
    nh     = P.N_ELM_hidden;
    T_per  = P.T_period;
    yte    = D.y_test;   n_test = D.n_test;
    denom  = {D.MAE_P, D.RMSE_P, D.RMCE_P, D.mean_y_test};

    R = table();

    % --- Persistance simple ---
    yP = D.Persis_simple_test;
    R = [R; ref_row('Persistence_P', 0, NaN, D, yP, denom)];

    % --- Persistance cyclique (recurrence du cycle diurne) ---
    yPc = zeros(n_test, 1);
    for k = 1:n_test
        idx_d = D.offset_base + k - T_per;
        if idx_d >= 1 && idx_d <= numel(D.data)
            yPc(k) = D.data(idx_d);
        else
            yPc(k) = yP(k);          % bord : repli sur la persistance simple
        end
    end
    R = [R; ref_row('Persistence_Pcyclic', 0, NaN, D, yPc, denom)];

    % --- Blend (poids par phase estime sur la calibration) ---
    data_tr = D.data(1 : D.idx_split + LB + FH);
    rho     = cyclic_correlation(data_tr, FH, T_per);
    yBL     = zeros(n_test, 1);
    for k = 1:n_test
        idx_d   = D.offset_base + k;
        phase   = mod(idx_d - 1, T_per) + 1;
        lambda  = max(0, min(1, 0.5 * (1 + rho(phase))));
        yBL(k)  = (1 - lambda) * yPc(k) + lambda * yP(k);
    end
    R = [R; ref_row('BLEND_tilde', 0, NaN, D, yBL, denom)];

    % --- ELM plein (48 retards + features temporelles) ---
    Ztr = [D.X_train, D.TEMP_train];
    Zte = [D.X_test,  D.TEMP_test];
    [beta, IW, Bias] = elm_train(Ztr, D.y_train, nh, P.N_ELM_candidates, P.ridge);
    yELM = 1 ./ (1 + exp(-(Zte * IW' + Bias'))) * beta;
    nP_e = nh * (LB + n_temp) + 2 * nh;
    R = [R; ref_row('ELM_full', nP_e, 0.0, D, yELM, denom, LB)];

    % --- AR plein (lineaire, moindres carres) ---
    beta_a = [Ztr, ones(size(Ztr,1),1)] \ D.y_train;
    yAR    = [Zte, ones(size(Zte,1),1)] * beta_a;
    nP_a   = LB + n_temp + 1;
    R = [R; ref_row('AR_full', nP_a, 100*(1 - nP_a/D.nParams_full), D, yAR, denom, LB)];
end

% -------------------------------------------------------------------------
function T = ref_row(name, nP, redPct, D, y_pred, denom, bestdim)
% Fabrique une ligne de reference en evaluant les metriques d'erreur.
    if nargin < 7, bestdim = NaN; end
    [rmse, nrmse, r2, n1, n2, n3, nS] = ...
        eval_metrics(D.y_test, y_pred, denom{1}, denom{2}, denom{3}, denom{4}, D.isnight, D.night);
    T = result_row(name, D.LB_days, D.FH_hours, bestdim, nP, redPct, NaN, ...
                   rmse, nrmse, r2, n1, n2, n3, nS);
end

% -------------------------------------------------------------------------
function rho = cyclic_correlation(x, FH, T)
% Correlation empirique, conditionnee a la phase (heure du jour), entre la
% production et celle FH pas plus tard. Sert de poids au blend.
    N   = numel(x);
    rho = zeros(T, 1);
    for k = 1:T
        idx = k:T:(N - FH);
        if numel(idx) < 3, rho(k) = 0; continue; end
        c = corrcoef(x(idx), x(idx + FH));
        v = c(1, 2);
        if isnan(v), v = 0; end
        rho(k) = v;
    end
end
