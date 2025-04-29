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

        private class RegressionDataPoint
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

                var labelPipeline = mlContext.Transforms.CopyColumns("Label", config.TargetField);
                var labeledData = labelPipeline.Fit(dataView).Transform(dataView);

                var dataPoints = mlContext.Data
                    .CreateEnumerable<RegressionDataPoint>(labeledData, reuseRowObject: false)
                    .ToList();

                var schemaDef = SchemaDefinition.Create(typeof(RegressionDataPoint));
                schemaDef[nameof(RegressionDataPoint.Features)].ColumnType = new VectorDataViewType(
                    NumberDataViewType.Single,
                    featureNames.Length);

                var typedData = mlContext.Data.LoadFromEnumerable(dataPoints, schemaDef);

                var split = mlContext.Data.TrainTestSplit(
                    typedData,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                var pipeline = mlContext.Transforms.NormalizeMinMax("Features")
                    .AppendCacheCheckpoint(mlContext)
                    .Append(trainer);

                var start = DateTime.Now;
                var model = await Task.Run(() => pipeline.Fit(split.TrainSet));
                Console.WriteLine($"Training completed in {(DateTime.Now - start).TotalSeconds:N1} seconds");

                var predictions = model.Transform(split.TestSet);
                var metrics = mlContext.Regression.Evaluate(predictions);
                ConsoleHelper.PrintRegressionMetrics(config.TrainingParameters.Algorithm, metrics);

                var modelPath = $"Regression_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, typedData.Schema, modelPath);
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
                Console.WriteLine($"\nError during regression training: {ex.Message}");
                throw;
            }
        }
    }
}