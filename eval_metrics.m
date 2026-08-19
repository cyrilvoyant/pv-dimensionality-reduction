function [rmse, nrmse, r2, nice1, nice2, nice3, niceS] = ...
        eval_metrics(y_test, y_pred, MAE_P, RMSE_P, RMCE_P, mean_y, isnight)
%   isnight (optionnel) : masque logique du test. La ou il vaut true, la
%   prevision est forcee a 0 (production PV nulle la nuit, rien a prevoir).
    if nargin >= 7 && ~isempty(isnight)
        y_pred(isnight) = 0;
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
