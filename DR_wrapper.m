function W = DR_wrapper(D, EMB, P)
% Analyse WRAPPER : pour chaque methode de reduction, on balaie les dimensions
% cibles, on couple l'espace reduit (augmente des features temporelles) a un
% operateur de prevision, et on retient la dimension qui minimise NICE_Sigma
% sur le test. Deux operateurs sont evalues cote a cote :
%   - ELM  (non lineaire)          -> ligne "<methode>"
%   - AR   (lineaire, moindres c.) -> ligne "<methode>_AR"
% La comparaison des deux mesure ce que la non-linearite apporte a dimension
% reduite egale.
%
% NB. La dimension optimale est choisie sur le meme jeu que celui rapporte
% (protocole a deux espaces, assume) : c'est un choix d'evaluation, discute
% dans le papier.
    techniques = P.techniques;
    nTech = numel(techniques);
    tdims = 2:P.LB;   nDim = numel(tdims);
    n_temp = D.n_temp;   nh = P.N_ELM_hidden;

    W = table();

    for i = 1:nTech
        E      = EMB(i);
        method = E.method;
        fprintf('    [wrapper] %s\n', method);

        % Variables locales pour le parfor (evite de diffuser toute la struct P).
        TEtr = D.TEMP_train;  TEte = D.TEMP_test;
        ytr  = D.y_train;     yte  = D.y_test;
        nc   = P.N_ELM_candidates;   ridge = P.ridge;
        MAE_P = D.MAE_P; RMSE_P = D.RMSE_P; RMCE_P = D.RMCE_P; myt = D.mean_y_test;
        nite  = D.isnight;   % masque nuit
        nmode = D.night;     % 'day' (defaut) | 'zero' | 'all'

        % ELM
        NS_e = nan(1,nDim); RMSE_e = nan(1,nDim); nRMSE_e = nan(1,nDim);
        R2_e = nan(1,nDim); N1_e = nan(1,nDim); N2_e = nan(1,nDim); N3_e = nan(1,nDim);
        % AR
        NS_a = nan(1,nDim); RMSE_a = nan(1,nDim); nRMSE_a = nan(1,nDim);
        R2_a = nan(1,nDim); N1_a = nan(1,nDim); N2_a = nan(1,nDim); N3_a = nan(1,nDim);

        Mw = P.parworkers;
        parfor (j = 1:nDim, Mw)
            [Ytr_d, Yte_d, ok] = emb_slice(E, j);
            if ~ok, continue; end

            Ztr = [Ytr_d, TEtr];      % espace reduit + features temporelles
            Zte = [Yte_d, TEte];

            % (a) ELM
            [beta, IW, Bias] = elm_train(Ztr, ytr, nh, nc, ridge);
            yhat = 1 ./ (1 + exp(-(Zte * IW' + Bias'))) * beta;
            [RMSE_e(j), nRMSE_e(j), R2_e(j), N1_e(j), N2_e(j), N3_e(j), NS_e(j)] = ...
                eval_metrics(yte, yhat, MAE_P, RMSE_P, RMCE_P, myt, nite, nmode);

            % (b) AR reduit (lineaire)
            ba   = [Ztr, ones(size(Ztr,1),1)] \ ytr;
            yhat = [Zte, ones(size(Zte,1),1)] * ba;
            [RMSE_a(j), nRMSE_a(j), R2_a(j), N1_a(j), N2_a(j), N3_a(j), NS_a(j)] = ...
                eval_metrics(yte, yhat, MAE_P, RMSE_P, RMCE_P, myt, nite, nmode);
        end

        % Meilleure dimension ELM
        [~, ib] = min(NS_e);   bd = tdims(ib);
        nP = nh * (bd + n_temp) + 2 * nh;
        W = [W; result_row(method, D.LB_days, D.FH_hours, bd, nP, ...
                 100*(1 - nP/D.nParams_full), NaN, ...
                 RMSE_e(ib), nRMSE_e(ib), R2_e(ib), N1_e(ib), N2_e(ib), N3_e(ib), NS_e(ib))];

        % Meilleure dimension AR reduit
        [~, ia] = min(NS_a);   bda = tdims(ia);
        nPa = bda + n_temp + 1;
        W = [W; result_row([method '_AR'], D.LB_days, D.FH_hours, bda, nPa, ...
                 100*(1 - nPa/D.nParams_full), NaN, ...
                 RMSE_a(ia), nRMSE_a(ia), R2_a(ia), N1_a(ia), N2_a(ia), N3_a(ia), NS_a(ia))];
    end
end
