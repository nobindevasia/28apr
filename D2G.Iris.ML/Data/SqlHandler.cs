using System;
using System.Data;
using System.Linq;
using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Interfaces;

namespace D2G.Iris.ML.Data
{
    public class SqlHandler : ISqlHandler
    {
        private SqlConnectionStringBuilder _builder;
        private readonly string _tableName;

        public SqlHandler(string tableName)
        {
            _tableName = tableName;
        }

        public void Connect(DatabaseConfig dbConfig)
        {
            _builder = new SqlConnectionStringBuilder()
            {
                DataSource = dbConfig.Server,
                InitialCatalog = dbConfig.Database,
                IntegratedSecurity = true,
                Pooling = true,
                TrustServerCertificate = true,
                ConnectTimeout = 60
            };
        }

        public string GetConnectionString()
        {
            if (_builder == null)
                throw new InvalidOperationException("Database connection not initialized");
            return _builder.ConnectionString;
        }

        public void SaveToSql(
            string tableName,
            IDataView processedData,
            string[] featureNames,
            string targetField,
            ModelType modelType)
        {
            using (var connection = new SqlConnection(GetConnectionString()))
            {
                connection.Open();

                string targetColumnType = modelType switch
                {
                    ModelType.BinaryClassification => "bit",
                    ModelType.MultiClassClassification => "int",
                    ModelType.Regression => "float",
                    _ => "float"
                };

                var schema = tableName.Split('.').First();
                var tableNameOnly = tableName.Split('.').Last();

                string createTableSql = $@"
                IF EXISTS (SELECT * FROM sys.tables WHERE name = '{tableNameOnly}' 
                          AND SCHEMA_NAME(schema_id) = '{schema}')
                BEGIN
                    DROP TABLE {tableName}
                END

                CREATE TABLE {tableName} (
                    {string.Join(",\n    ", featureNames.Select(f => $"[{f}] float"))},
                    [{targetField}] {targetColumnType},
                    ProcessedDateTime datetime DEFAULT GETDATE()
                )";

                using (var command = new SqlCommand(createTableSql, connection))
                {
                    try
                    {
                        command.ExecuteNonQuery();
                        Console.WriteLine($"\nTable {tableName} dropped and recreated successfully");
                    }
                    catch (SqlException ex)
                    {
                        Console.WriteLine($"Error recreating table: {ex.Message}");
                        throw;
                    }
                }

                using (var bulkCopy = new SqlBulkCopy(connection))
                {
                    bulkCopy.DestinationTableName = tableName;
                    bulkCopy.BatchSize = 1000;
                    bulkCopy.BulkCopyTimeout = 600;

                    var dataTable = new DataTable();
                    foreach (var feature in featureNames)
                    {
                        dataTable.Columns.Add(feature, typeof(float));
                    }

                    Type targetColumnClrType = modelType switch
                    {
                        ModelType.BinaryClassification => typeof(bool),
                        ModelType.MultiClassClassification => typeof(int),
                        ModelType.Regression => typeof(float),
                        _ => typeof(float)
                    };
                    dataTable.Columns.Add(targetField, targetColumnClrType);

                    var targetColInfo = processedData.Schema.GetColumnOrNull(targetField);
                    bool labelIsBoolean = false;
                    bool labelIsLong = false;

                    if (targetColInfo.HasValue)
                    {
                        labelIsBoolean = targetColInfo.Value.Type == BooleanDataViewType.Instance;
                        labelIsLong = targetColInfo.Value.Type == NumberDataViewType.Int64;
                    }

                    bool useFeatureVector = false;
                    var featuresColumn = processedData.Schema.GetColumnOrNull("Features");

                    if (featuresColumn.HasValue && featuresColumn.Value.Type is VectorDataViewType)
                    {
                        bool hasSomeIndividualFeatures = false;
                        foreach (var feature in featureNames)
                        {
                            if (processedData.Schema.GetColumnOrNull(feature).HasValue)
                            {
                                hasSomeIndividualFeatures = true;
                                break;
                            }
                        }

                        useFeatureVector = !hasSomeIndividualFeatures;
                    }

                    using (var cursor = processedData.GetRowCursor(processedData.Schema))
                    {
                        while (cursor.MoveNext())
                        {
                            var row = dataTable.NewRow();

                            if (useFeatureVector)
                            {
                                var featuresGetter = cursor.GetGetter<VBuffer<float>>(featuresColumn.Value);
                                VBuffer<float> features = default;
                                featuresGetter(ref features);

                                float[] featureValues = features.GetValues().ToArray();
                                if (features.IsDense && featureValues.Length >= featureNames.Length)
                                {
                                    for (int i = 0; i < featureNames.Length; i++)
                                    {
                                        row[featureNames[i]] = featureValues[i];
                                    }
                                }
                                else
                                {
                                    Console.WriteLine("Warning: Feature vector is sparse or has wrong size. " +
                                                     $"Expected {featureNames.Length}, got {featureValues.Length}");
                                }
                            }
                            else
                            {
                                for (int i = 0; i < featureNames.Length; i++)
                                {
                                    var featureColumn = processedData.Schema.GetColumnOrNull(featureNames[i]);
                                    if (featureColumn.HasValue)
                                    {
                                        var getter = cursor.GetGetter<float>(featureColumn.Value);
                                        float value = 0;
                                        getter(ref value);
                                        row[featureNames[i]] = value;
                                    }
                                    else
                                    {
                                        row[featureNames[i]] = DBNull.Value;
                                        Console.WriteLine($"Warning: Feature {featureNames[i]} not found in schema");
                                    }
                                }
                            }

                            var targetColumn = processedData.Schema.GetColumnOrNull(targetField);
                            if (targetColumn.HasValue)
                            {
                                switch (modelType)
                                {
                                    case ModelType.BinaryClassification:
                                        if (labelIsBoolean)
                                        {

                                            var binaryGetter = cursor.GetGetter<bool>(targetColumn.Value);
                                            bool binaryValue = false;
                                            binaryGetter(ref binaryValue);
                                            row[targetField] = binaryValue;
                                        }
                                        else if (labelIsLong)
                                        {
                                            var binaryLongGetter = cursor.GetGetter<long>(targetColumn.Value);
                                            long binaryLongValue = 0;
                                            binaryLongGetter(ref binaryLongValue);
                                            row[targetField] = binaryLongValue > 0;
                                        }
                                        else
                                        {
                                            var binaryFloatGetter = cursor.GetGetter<float>(targetColumn.Value);
                                            float binaryFloatValue = 0;
                                            binaryFloatGetter(ref binaryFloatValue);
                                            row[targetField] = binaryFloatValue > 0;
                                        }
                                        break;

                                    case ModelType.MultiClassClassification:
                                        var multiClassGetter = cursor.GetGetter<long>(targetColumn.Value);
                                        long multiClassValue = 0;
                                        multiClassGetter(ref multiClassValue);
                                        row[targetField] = (int)multiClassValue;
                                        break;

                                    case ModelType.Regression:
                                        var regressionGetter = cursor.GetGetter<float>(targetColumn.Value);
                                        float regressionValue = 0;
                                        regressionGetter(ref regressionValue);
                                        row[targetField] = regressionValue;
                                        break;
                                }
                            }

                            dataTable.Rows.Add(row);
                        }
                    }

                    foreach (DataColumn column in dataTable.Columns)
                    {
                        bulkCopy.ColumnMappings.Add(column.ColumnName, column.ColumnName);
                    }

                    try
                    {
                        bulkCopy.WriteToServer(dataTable);
                        Console.WriteLine($"Successfully inserted {dataTable.Rows.Count} rows into {tableName}");
                    }
                    catch (Exception ex)
                    {
                        throw new Exception($"Failed to insert data into table {tableName}: {ex.Message}", ex);
                    }
                }
            }
        }
    }
}