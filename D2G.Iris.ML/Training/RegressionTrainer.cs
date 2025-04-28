using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.Training
{
    public class RegressionTrainer : IModelTrainer
    {
        private readonly MLContext _mlContext;
        private readonly TrainerFactory _trainerFactory;

        public RegressionTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext;
            _trainerFactory = trainerFactory;
        }

        private class ModelInput
        {
            [VectorType]
            public float[] Features { get; set; }
            public float Label { get; set; }
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

                // Copy column transformation for the label
                var labelPipeline = mlContext.Transforms.CopyColumns("Label", config.TargetField);
                var labeledData = labelPipeline.Fit(dataView).Transform(dataView);

                // Check if we have the 'Features' column already - this happens with PCA feature selection
                bool hasPrecomputedFeatures = labeledData.Schema.GetColumnOrNull("Features").HasValue;
                IDataView transformedData;

                if (!hasPrecomputedFeatures)
                {
                    Console.WriteLine("Creating feature vector from individual feature columns...");
                    // Create the feature vector from individual columns
                    var featurePipeline = mlContext.Transforms.Concatenate("Features", featureNames);
                    transformedData = featurePipeline.Fit(labeledData).Transform(labeledData);
                }
                else
                {
                    Console.WriteLine("Using precomputed feature vector...");
                    transformedData = labeledData;
                }

                // Print schema to debug
                Console.WriteLine("\nData schema after transformation:");
                var previewCount = Math.Min(5, (int)(transformedData.GetRowCount() ?? 0));
                if (previewCount > 0)
                {
                    var featuresType = transformedData.Schema["Features"].Type;
                    Console.WriteLine($"Features column type: {featuresType}");

                    if (featuresType is VectorDataViewType vectorType)
                    {
                        Console.WriteLine($"Features vector size: {vectorType.Size}");
                    }
                }

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

                // Save model
                var modelPath = $"Regression_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, transformedData.Schema, modelPath);
                Console.WriteLine($"\nModel saved to: {modelPath}");

                // Save model info
                await ModelHelper.CreateModelInfo<RegressionMetrics, float>(
                    metrics,
                    dataView,
                    featureNames,
                    config,
                    processedData);

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