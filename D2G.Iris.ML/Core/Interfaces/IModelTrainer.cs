using D2G.Iris.ML.Core.Models;
using Microsoft.ML;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IModelTrainer
    {
        Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData);
    }
}