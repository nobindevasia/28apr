using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
//using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
//using D2G.Iris.ML.Utils;
using D2G.Iris.ML.Training;

namespace D2G.Iris.ML.Training
{
    public abstract class BaseModelTrainer
    {
        protected readonly MLContext _mlContext;
        protected readonly TrainerFactory _trainerFactory;

        protected BaseModelTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext;
            _trainerFactory = trainerFactory;
        }

        public abstract Task<IDataView> PrepareDataView(
            MLContext mlContext,
            ProcessedData processedData,
            string[] featureNames,
            string targetField);

        public abstract Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData);

        protected IEstimator<ITransformer> GetBasePipeline(MLContext mlContext)
        {
            return mlContext.Transforms.NormalizeMinMax("Features")
                           .AppendCacheCheckpoint(mlContext);
        }
    }
}