/* Общие подписи для обзора и реестра; исходные ID остаются доступны в деталях. */
window.GenesisModels = (() => {
  const types = {logistic:'Logistic regression',xgboost:'XGBoost',random_forest:'Random forest',hist_gradient_boosting:'Gradient boosting',svm:'SVM',knn:'K-nearest neighbors',mlp:'Neural network'};
  /** @param {{name?:string,target?:string,model_type?:string}} model @returns {string} */
  function label(model) {
    if (model.name?.trim()) return model.name.trim();
    const target = String(model.target || 'Model').replaceAll('_',' ');
    const title = target.charAt(0).toUpperCase() + target.slice(1);
    return title + ' · ' + (types[model.model_type] || String(model.model_type || 'Trained model').replaceAll('_',' '));
  }
  /** @param {{model_id:string,version?:number,created_at?:string}} model @returns {string} */
  function identity(model) {
    const suffix = model.model_id.split('_').slice(-2).join('_');
    return (model.version ? 'v' + model.version + ' · ' : '') + suffix;
  }
  /** @param {{metrics?:Object<string,unknown>}} model @returns {{label:string,value:string}} */
  function metric(model) {
    const values = model.metrics || {};
    const key = ['accuracy_test','r2_test','f1_test','accuracy','r2','f1'].find(k => typeof values[k] === 'number' && Number.isFinite(values[k]));
    if (!key) return {label:'No score',value:'—'};
    const names = {accuracy_test:'Test accuracy',r2_test:'Test R²',f1_test:'Test F1',accuracy:'Accuracy',r2:'R²',f1:'F1'};
    return {label:names[key],value:Number(values[key]).toFixed(3)};
  }
  return {label,identity,metric};
})();
