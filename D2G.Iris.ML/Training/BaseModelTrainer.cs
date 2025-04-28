using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public abstract class BaseModelTrainer : IModelTrainer
    {
        protected readonly MLContext _mlContext;
        protected readonly TrainerFactory _trainerFactory;

        protected BaseModelTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext;
            _trainerFactory = trainerFactory;
        }

        /// <summary>
        /// Prepares the data for training with appropriate schema definition
        /// </summary>
        protected abstract Task<IDataView> PrepareDataForTraining(
            IDataView dataView,
            string[] featureNames);

        /// <summary>
        /// Gets the appropriate trainer based on model type and configuration
        /// </summary>
        protected abstract IEstimator<ITransformer> GetTrainer(ModelConfig config);

        /// <summary>
        /// Evaluates the model and returns metrics
        /// </summary>
        protected abstract object EvaluateModel(
            ITransformer model,
            IDataView testData);

        /// <summary>
        /// Trains a model using the provided data and configuration
        /// </summary>
   

        /// <summary>
        /// Prints metrics and saves model information
        /// </summary>
        protected virtual async Task PrintMetricsAndSaveInfo(
            object metrics,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            switch (config.ModelType)
            {
                case Core.Enums.ModelType.BinaryClassification:
                    if (metrics is BinaryClassificationMetrics binaryMetrics)
                    {
                        ConsoleHelper.PrintBinaryClassificationMetrics(config.TrainingParameters.Algorithm, binaryMetrics);
                        await ModelHelper.CreateModelInfo<BinaryClassificationMetrics, float>(
                            binaryMetrics, dataView, featureNames, config, processedData);
                    }
                    break;

                case Core.Enums.ModelType.MultiClassClassification:
                    if (metrics is MulticlassClassificationMetrics multiclassMetrics)
                    {
                        ConsoleHelper.PrintMultiClassClassificationMetrics(config.TrainingParameters.Algorithm, multiclassMetrics);
                        await ModelHelper.CreateModelInfo<MulticlassClassificationMetrics, float>(
                            multiclassMetrics, dataView, featureNames, config, processedData);
                    }
                    break;

                case Core.Enums.ModelType.Regression:
                    if (metrics is RegressionMetrics regressionMetrics)
                    {
                        ConsoleHelper.PrintRegressionMetrics(config.TrainingParameters.Algorithm, regressionMetrics);
                        await ModelHelper.CreateModelInfo<RegressionMetrics, float>(
                            regressionMetrics, dataView, featureNames, config, processedData);
                    }
                    break;
            }
        }

   
    }
}