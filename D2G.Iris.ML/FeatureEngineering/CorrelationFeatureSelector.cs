using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using MathNet.Numerics.Statistics;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class CorrelationFeatureSelector : IFeatureSelector
    {
        private readonly MLContext _mlContext;
        private readonly StringBuilder _report;

        public CorrelationFeatureSelector(MLContext mlContext)
        {
            _mlContext = mlContext;
            _report = new StringBuilder();
        }

        private class RegressionFeatureRow
        {
            [VectorType]
            public float[] Features { get; set; }
            public float RegressionTarget { get; set; }
        }

        private class ClassificationFeatureRow
        {
            [VectorType]
            public float[] Features { get; set; }
            public long ClassificationTarget { get; set; }
        }

        public async Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            _report.Clear();
            _report.AppendLine("\nCorrelation-based Feature Selection Results:");
            _report.AppendLine("----------------------------------------------");

            try
            {
                // First, create a Features column if it doesn't exist
                IDataView featuresData = data;
                if (data.Schema.GetColumnOrNull("Features") == null)
                {
                    var featuresPipeline = mlContext.Transforms.Concatenate("Features", candidateFeatures);
                    featuresData = featuresPipeline.Fit(data).Transform(data);
                }

                // Debug schema
                _report.AppendLine("\nData schema:");
                foreach (var col in featuresData.Schema)
                {
                    _report.AppendLine($"  Column: {col.Name}, Type: {col.Type}");
                }

                // Check if target field exists
                if (!featuresData.Schema.GetColumnOrNull(targetField).HasValue)
                {
                    _report.AppendLine($"Error: Target field '{targetField}' not found in schema");
                    throw new InvalidOperationException($"Target field '{targetField}' not found");
                }

                // Copy target field to a standard name column for extraction
                IDataView renamedData;
                if (modelType == ModelType.Regression)
                {
                    var renamePipeline = mlContext.Transforms.CopyColumns("RegressionTarget", targetField);
                    renamedData = renamePipeline.Fit(featuresData).Transform(featuresData);
                }
                else
                {
                    var renamePipeline = mlContext.Transforms.CopyColumns("ClassificationTarget", targetField);
                    renamedData = renamePipeline.Fit(featuresData).Transform(featuresData);
                }

                // Extract feature and target data
                List<double> targetValues = new List<double>();
                List<float[]> featureVectors = new List<float[]>();

                if (modelType == ModelType.Regression)
                {
                    var rows = mlContext.Data.CreateEnumerable<RegressionFeatureRow>(renamedData, reuseRowObject: false).ToList();
                    if (rows.Count == 0)
                    {
                        throw new InvalidOperationException("No rows found in data");
                    }

                    foreach (var row in rows)
                    {
                        featureVectors.Add(row.Features);
                        targetValues.Add(row.RegressionTarget);
                    }
                }
                else
                {
                    var rows = mlContext.Data.CreateEnumerable<ClassificationFeatureRow>(renamedData, reuseRowObject: false).ToList();
                    if (rows.Count == 0)
                    {
                        throw new InvalidOperationException("No rows found in data");
                    }

                    foreach (var row in rows)
                    {
                        featureVectors.Add(row.Features);
                        targetValues.Add(row.ClassificationTarget);
                    }
                }

                _report.AppendLine($"\nExtracted {featureVectors.Count} samples for correlation analysis");

                // Calculate correlations with target for each feature
                var targetCorrelations = new Dictionary<string, double>();

                for (int i = 0; i < candidateFeatures.Length && i < featureVectors[0].Length; i++)
                {
                    try
                    {
                        var featureColumn = featureVectors.Select(f => (double)f[i]).ToArray();
                        var correlation = Math.Abs(Correlation.Pearson(featureColumn, targetValues.ToArray()));
                        targetCorrelations[candidateFeatures[i]] = correlation;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error calculating correlation for {candidateFeatures[i]}: {ex.Message}");
                        targetCorrelations[candidateFeatures[i]] = 0; // Default to zero correlation on error
                    }
                }

                // Sort features by correlation
                var sortedFeatures = targetCorrelations
                    .OrderByDescending(x => x.Value)
                    .ToList();

                _report.AppendLine("\nFeatures Ranked by Target Correlation:");
                foreach (var pair in sortedFeatures)
                {
                    _report.AppendLine($"{pair.Key,-40} | {pair.Value:F4}");
                }

                // Select features considering multicollinearity
                var selectedFeatures = new List<string>();
                var selectedIndices = new List<int>();

                foreach (var pair in sortedFeatures)
                {
                    if (selectedFeatures.Count >= config.MaxFeatures)
                        break;

                    var currentIndex = Array.IndexOf(candidateFeatures, pair.Key);
                    var currentColumn = featureVectors.Select(f => (double)f[currentIndex]).ToArray();

                    bool isHighlyCorrelated = false;
                    foreach (var selectedIndex in selectedIndices)
                    {
                        var selectedColumn = featureVectors.Select(f => (double)f[selectedIndex]).ToArray();
                        try
                        {
                            var correlation = Math.Abs(Correlation.Pearson(currentColumn, selectedColumn));
                            if (correlation > config.MulticollinearityThreshold)
                            {
                                isHighlyCorrelated = true;
                                break;
                            }
                        }
                        catch
                        {
                            // Skip this comparison if correlation calculation fails
                        }
                    }

                    if (!isHighlyCorrelated)
                    {
                        selectedFeatures.Add(pair.Key);
                        selectedIndices.Add(currentIndex);
                    }
                }

                // Ensure we have at least one feature
                if (selectedFeatures.Count == 0 && sortedFeatures.Count > 0)
                {
                    selectedFeatures.Add(sortedFeatures[0].Key);
                }

                _report.AppendLine($"\nSelection Summary:");
                _report.AppendLine($"Original features: {candidateFeatures.Length}");
                _report.AppendLine($"Selected features: {selectedFeatures.Count}");
                _report.AppendLine($"Multicollinearity threshold: {config.MulticollinearityThreshold}");
                _report.AppendLine("\nSelected Features:");
                foreach (var feature in selectedFeatures)
                {
                    _report.AppendLine($"- {feature} (correlation with target: {targetCorrelations[feature]:F4})");
                }

                // Create transformed data with selected features
                var pipeline = mlContext.Transforms.Concatenate("Features", selectedFeatures.ToArray());
                var transformedData = pipeline.Fit(data).Transform(data);

                return (transformedData, selectedFeatures.ToArray(), _report.ToString());
            }
            catch (Exception ex)
            {
                _report.AppendLine($"Error during correlation analysis: {ex.Message}");
                Console.WriteLine($"Full error details: {ex}");
                throw;
            }
        }
    }
}