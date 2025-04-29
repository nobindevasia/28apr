using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
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
                // Extract feature values
                var featureValues = new List<double[]>();
                foreach (var feature in candidateFeatures)
                {
                    // Extract column values and convert to double[]
                    featureValues.Add(GetColumnValues(data, feature));
                }

                // Extract target values
                var targetValues = GetColumnValues(data, targetField);

                // Calculate target correlations for each feature
                var targetCorrelations = new Dictionary<string, double>();
                for (int i = 0; i < candidateFeatures.Length; i++)
                {
                    var correlation = Correlation.Pearson(featureValues[i], targetValues);
                    targetCorrelations[candidateFeatures[i]] = Math.Abs(correlation);
                }

                // Calculate correlation matrix between features
                var correlationMatrix = Matrix<double>.Build.Dense(
                    candidateFeatures.Length,
                    candidateFeatures.Length
                );

                for (int i = 0; i < candidateFeatures.Length; i++)
                {
                    for (int j = 0; j < candidateFeatures.Length; j++)
                    {
                        correlationMatrix[i, j] = Correlation.Pearson(
                            featureValues[i],
                            featureValues[j]
                        );
                    }
                }

                // Sort features by target correlation
                var sortedFeatures = targetCorrelations
                    .OrderByDescending(x => x.Value)
                    .Select(x => x.Key)
                    .ToList();

                _report.AppendLine("\nFeatures Ranked by Target Correlation:");
                foreach (var feature in sortedFeatures)
                {
                    _report.AppendLine($"{feature,-40} | {targetCorrelations[feature]:F4}");
                }

                // Select features based on correlation and multicollinearity
                var selectedFeatures = new List<string>();
                foreach (var feature in sortedFeatures)
                {
                    if (selectedFeatures.Count >= config.MaxFeatures)
                        break;

                    bool isHighlyCorrelated = false;
                    foreach (var selectedFeature in selectedFeatures)
                    {
                        var i1 = Array.IndexOf(candidateFeatures, feature);
                        var i2 = Array.IndexOf(candidateFeatures, selectedFeature);
                        if (Math.Abs(correlationMatrix[i1, i2]) > config.MulticollinearityThreshold)
                        {
                            isHighlyCorrelated = true;
                            break;
                        }
                    }

                    if (!isHighlyCorrelated)
                    {
                        selectedFeatures.Add(feature);
                    }
                }

                // Ensure we have at least one feature
                if (selectedFeatures.Count == 0 && sortedFeatures.Count > 0)
                {
                    selectedFeatures.Add(sortedFeatures[0]);
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

        private double[] GetColumnValues(IDataView dataView, string columnName)
        {
            var column = dataView.Schema.GetColumnOrNull(columnName);
            if (!column.HasValue)
                throw new ArgumentException($"Column '{columnName}' not found in data");

            var type = column.Value.Type;

            if (type is NumberDataViewType numType)
            {
                if (numType.RawType == typeof(float))
                    return dataView.GetColumn<float>(columnName).Select(v => (double)v).ToArray();
                else if (numType.RawType == typeof(double))
                    return dataView.GetColumn<double>(columnName).ToArray();
                else if (numType.RawType == typeof(int))
                    return dataView.GetColumn<int>(columnName).Select(v => (double)v).ToArray();
                else if (numType.RawType == typeof(long))
                    return dataView.GetColumn<long>(columnName).Select(v => (double)v).ToArray();
                else
                    return dataView.GetColumn<float>(columnName).Select(v => (double)v).ToArray();
            }
            else if (type is BooleanDataViewType)
            {
                return dataView.GetColumn<bool>(columnName).Select(v => v ? 1.0 : 0.0).ToArray();
            }

            throw new NotSupportedException($"Column type {type} is not supported for correlation analysis");
        }
    }
}