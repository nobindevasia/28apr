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

                var labelPipeline = mlContext.Transforms.CopyColumns("Label", config.TargetField);
                var labeledData = labelPipeline.Fit(dataView).Transform(dataView);

                bool hasPrecomputedFeatures = labeledData.Schema.GetColumnOrNull("Features").HasValue;
                IDataView transformedData;

                if (!hasPrecomputedFeatures)
                {

                    var featurePipeline = mlContext.Transforms.Concatenate("Features", featureNames);
                    transformedData = featurePipeline.Fit(labeledData).Transform(labeledData);
                }
                else
                {
                    Console.WriteLine("Using precomputed feature vector...");
                    transformedData = labeledData;
                }


                //Console.WriteLine("\nData schema after transformation:");
                //var previewCount = Math.Min(5, (int)(transformedData.GetRowCount() ?? 0));
                //if (previewCount > 0)
                //{
                //    var featuresType = transformedData.Schema["Features"].Type;
                //    Console.WriteLine($"Features column type: {featuresType}");

                //    if (featuresType is VectorDataViewType vectorType)
                //    {
                //        Console.WriteLine($"Features vector size: {vectorType.Size}");
                //    }
                //}

                var splitData = mlContext.Data.TrainTestSplit(
                    transformedData,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);


                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                var pipeline = mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .AppendCacheCheckpoint(mlContext)
                    .Append(trainer);


                var trainStartTime = DateTime.Now;
                var model = await Task.Run(() => pipeline.Fit(splitData.TrainSet));
                var trainingTime = DateTime.Now - trainStartTime;
                Console.WriteLine($"Training completed in {trainingTime.TotalSeconds:N1} seconds");


                var predictions = model.Transform(splitData.TestSet);
                var metrics = mlContext.Regression.Evaluate(predictions);

                ConsoleHelper.PrintRegressionMetrics(config.TrainingParameters.Algorithm, metrics);

                var modelPath = $"Regression_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, transformedData.Schema, modelPath);
                Console.WriteLine($"\nModel saved to: {modelPath}");

             
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