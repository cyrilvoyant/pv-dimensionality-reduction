function [rmse, nrmse, r2, nice1, nice2, nice3, niceS] = ...
        eval_metrics(y_test, y_pred, MAE_P, RMSE_P, RMCE_P, mean_y, isnight, night_mode)
% Traitement de la nuit (production PV nulle, rien a prevoir) selon night_mode :
%   'day'  : metriques calculees de JOUR uniquement, nuit exclue   [defaut]
%   'zero' : prevision forcee a 0 la nuit, metriques sur tout
%   'all'  : aucun traitement (jour + nuit bruts)
% isnight est le masque logique nuit du test.
    if nargin < 8 || isempty(night_mode), night_mode = 'day'; end
    if nargin >= 7 && ~isempty(isnight)
        switch night_mode
            case 'day'
                keep   = ~isnight;
                y_pred = y_pred(keep);
                y_test = y_test(keep);
            case 'zero'
                y_pred(isnight) = 0;
        end
    end
% Bloc de metriques commun a tous les modeles (baselines, ELM, AR).
%   nRMSE = RMSE / moyenne de production sur le test.
%   NICE^k = norme L^k de l'erreur du modele rapportee a celle de la
%   persistance simple (voir Voyant et al., NICEk). Par construction la
%   persistance a NICE^k = 1 ; un bon modele est < 1.
%   niceS = moyenne des trois ordres (critere de selection du wrapper).
    err   = y_pred(:) - y_test(:);
    rmse  = sqrt(mean(err.^2));
    nrmse = rmse / mean_y;
    r2    = 1 - sum(err.^2) / sum((y_test(:) - mean(y_test(:))).^2);

    nice1 = compute_Lk_error(y_test, y_pred, 1) / MAE_P;
    nice2 = compute_Lk_error(y_test, y_pred, 2) / RMSE_P;
    nice3 = compute_Lk_error(y_test, y_pred, 3) / RMCE_P;
    niceS = (nice1 + nice2 + nice3) / 3;
end
