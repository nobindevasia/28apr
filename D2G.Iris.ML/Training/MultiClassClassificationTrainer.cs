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
    public class MultiClassClassificationTrainer : IModelTrainer
    {
        private readonly MLContext _mlContext;
        private readonly TrainerFactory _trainerFactory;

        public MultiClassClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext;
            _trainerFactory = trainerFactory;
        }
        private class ModelInput
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }

        public async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting multiclass classification model training using {config.TrainingParameters.Algorithm}...");
            try
            {
                var data = mlContext.Data
                    .CreateEnumerable<ModelInput>(dataView, reuseRowObject: false)
                    .Select(row => new ModelInput
                    {
                        Features = row.Features,
                        Label = row.Label
                    })
                    .ToList();

                var schema = SchemaDefinition.Create(typeof(ModelInput));
                schema["Features"].ColumnType =
                    new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);

                var fixedData = mlContext.Data.LoadFromEnumerable(data, schema);

                var splitData = mlContext.Data.TrainTestSplit(
                    fixedData,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                IEstimator<ITransformer> pipeline = mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .Append(mlContext.Transforms.Conversion
                        .MapValueToKey(outputColumnName: "Label", inputColumnName: "Label"))
                    .AppendCacheCheckpoint(mlContext);

                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                pipeline = pipeline
                    .Append(trainer)
                    .Append(mlContext.Transforms.Conversion
                        .MapKeyToValue("PredictedLabel", "PredictedLabel"));

                var trainStartTime = DateTime.Now;
                var model = await Task.Run(() => pipeline.Fit(splitData.TrainSet));
                var trainingTime = DateTime.Now - trainStartTime;
                Console.WriteLine($"Training completed in {trainingTime.TotalSeconds:N1} seconds");


                var predictions = model.Transform(splitData.TestSet);
                var metrics = mlContext.MulticlassClassification.Evaluate(
                    predictions);


                ConsoleHelper.PrintMultiClassClassificationMetrics(config.TrainingParameters.Algorithm, metrics);
                Console.WriteLine($"Confusion Matrix:\n{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");

                await ModelHelper.CreateModelInfo<MulticlassClassificationMetrics, float>(
                metrics,
                dataView,
                featureNames,
                config,
                processedData);

                var modelPath = $"MultiClassClassification_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, fixedData.Schema, modelPath);
                Console.WriteLine($"\nModel saved to: {modelPath}");

                return model;
            }

            catch (Exception ex)
            {
                Console.WriteLine($"\nError during model training: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                if (ex.InnerException != null)
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                throw;
            }
        }
    }
}