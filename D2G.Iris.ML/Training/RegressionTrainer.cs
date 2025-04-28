using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public class RegressionTrainer
    {
        private readonly MLContext _mlContext;
        private readonly TrainerFactory _trainerFactory;

        public RegressionTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext;
            _trainerFactory = trainerFactory;
        }

        public async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            try
            {
                Console.WriteLine($"\nStarting regression model training using {config.TrainingParameters.Algorithm}...");
                Console.WriteLine($"Selected features: {string.Join(", ", featureNames)}");

                // Create initial pipeline to extract features and label
                var dataPipeline = mlContext.Transforms
                    .CopyColumns("Label", config.TargetField)
                    .Append(mlContext.Transforms.Concatenate("Features", featureNames));

                var transformedData = dataPipeline.Fit(dataView).Transform(dataView);

                // Split data
                Console.WriteLine("\nSplitting data into training and test sets...");
                var splitData = mlContext.Data.TrainTestSplit(
                    transformedData,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                Console.WriteLine($"Training set size: {splitData.TrainSet.GetRowCount():N0} rows");
                Console.WriteLine($"Test set size: {splitData.TestSet.GetRowCount():N0} rows");

                // Create training pipeline
                Console.WriteLine("\nConfiguring training pipeline...");
                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                var pipeline = mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .AppendCacheCheckpoint(mlContext)
                    .Append(trainer);

                // Train model with progress reporting
                Console.WriteLine("\nTraining model...");
                var trainStartTime = DateTime.Now;
                var model = await Task.Run(() => pipeline.Fit(splitData.TrainSet));
                var trainingTime = DateTime.Now - trainStartTime;
                Console.WriteLine($"Training completed in {trainingTime.TotalSeconds:N1} seconds");

                // Evaluate model
                Console.WriteLine("\nEvaluating model...");
                var predictions = model.Transform(splitData.TestSet);
                var metrics = mlContext.Regression.Evaluate(predictions);

                // Print metrics using ConsoleHelper
                ConsoleHelper.PrintRegressionMetrics(config.TrainingParameters.Algorithm, metrics);

                // Save model information using ModelHelper
                await ModelHelper.CreateModelInfo<RegressionMetrics, float>(
                metrics,
                dataView,
                featureNames,
                config,
                processedData);
                // Save model
                var modelPath = $"Regression_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, transformedData.Schema, modelPath);
                Console.WriteLine($"\nModel saved to: {modelPath}");

                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError during model training: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                }
                throw;
            }
        }
    }
}