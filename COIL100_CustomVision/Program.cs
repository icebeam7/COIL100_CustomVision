using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction;
using System.Net.Http;
using System.Text;

namespace COIL100_CustomVision
{
	class Program
	{
		public static async Task Main()
		{
			// These are the image tags
			var labels = new Dictionary<int, string>()
			{
				{ 1,  "dristan cold box" },
				{ 2,  "onion" },
				{ 4,  "tomato" },
				{ 5,  "rolaids bottle" },
				{ 7,  "arizona iced tea" },
				{ 10, "cup" },
				{ 14, "cat" },
				{ 19, "firetruck car" },
				{ 28, "frog" },
				{ 31, "tylenol" },
				{ 33, "glue" },
				{ 35, "porcelain plate" },
				{ 37, "toy tank" },
				{ 46, "marlboro cigarette box" },
				{ 47, "donut toy" },
				{ 48, "piggy bank" },
				{ 49, "canada dry ginger ale" }
			};

			// Project name must match the one in Custom Vision
			var projectName = "COIL100 Small";

			// Replace with your own from Custom Vision project
			var endpoint = "";
			var customVisionKey = "";
			var resourceId = "";

			var publishedModelName = "coil100Model";

			if (string.IsNullOrWhiteSpace(endpoint) ||
				string.IsNullOrWhiteSpace(customVisionKey) ||
				string.IsNullOrWhiteSpace(resourceId))
			{
				Console.WriteLine("You need to set the endpoint, key and resource id. The program will end;");
				Console.ReadKey();
				return;
			}

			// Training Client
			var trainingApi = new CustomVisionTrainingClient()
			{
				ApiKey = customVisionKey,
				Endpoint = endpoint
			};

			Console.WriteLine($"Selecting existing project: {projectName}...");

			var projects = await trainingApi.GetProjectsAsync();
			var project = projects.FirstOrDefault(x => x.Name == projectName);

			if (project != null)
            {
				Console.WriteLine($"{projectName} found in Custom Vision workspace.");
                WriteSeparator();
			}
			else
			{
				Console.WriteLine($"Project {projectName} was not found in your subscription. The program will end.");
				return;
			}

			Console.WriteLine("Retrieving tags...");
			// Retrieve the tags that already exist in the project
			var modelTags = await trainingApi.GetTagsAsync(project.Id);
			var tags = new List<Tag>();
			var onlineImages = 0;

			foreach (var label in labels)
			{
				// Check if the label already exists
				var tag = modelTags.FirstOrDefault(x => x.Name == label.Value);

				if (tag == null)
				{
					// If not, create it
					tag = await trainingApi.CreateTagAsync(project.Id, label.Value);
					Console.WriteLine($"Tag {tag.Name} was created.");
				}
				else
				{
					// If so, just count images with this tag
					onlineImages += tag.ImageCount;
					Console.WriteLine($"Tag {label.Value} was NOT created (it already exists)");
				}

				tags.Add(tag);
			}

			WriteSeparator();

			var uploadImages = true;

			if (onlineImages > 0)
			{
				Console.WriteLine($"There are {onlineImages} training images already uploaded. Do you want to upload more? (Y/N)");
				uploadImages = Console.ReadKey().Key == ConsoleKey.Y;
			}

			Iteration iteration = null;

			if (uploadImages)
			{
				Console.WriteLine("\tUploading images");
				var images = LoadImagesFromDisk("images");

				foreach (var image in images)
				{
					var fileName = Path.GetFileName(image);
					var objImage = fileName.Split(new string[] { "__" }, StringSplitOptions.None);
					var id = int.Parse(objImage[0].Remove(0, 3));
					var label = labels.SingleOrDefault(t => t.Key == id);
					var tag = tags.Single(x => x.Name == label.Value);

					using (var stream = new MemoryStream(File.ReadAllBytes(image)))
					{
						await trainingApi.CreateImagesFromDataAsync(
							project.Id,
							stream,
							new List<Guid>() { tag.Id });

						Console.WriteLine($"Image {fileName} uploaded");
					}
				}

				WriteSeparator();

				try
				{
					// Now there are images with tags start training the project
					Console.WriteLine("\tTraining started...");
					iteration = await trainingApi.TrainProjectAsync(project.Id);

					// The returned iteration will be in progress, and can be queried periodically to see when it has completed
					while (iteration.Status == "Training")
					{
						Thread.Sleep(1000);
						Console.WriteLine($"Iteration '{iteration.Name}' status: {iteration.Status}");

						// Re-query the iteration to get it's updated status
						iteration = await trainingApi.GetIterationAsync(project.Id, iteration.Id);
					}

					Console.WriteLine($"Iteration '{iteration.Name}' status: {iteration.Status}");
					WriteSeparator();

					// The iteration is now trained. Publish it to the prediction endpoint.
					await trainingApi.PublishIterationAsync(
						project.Id,
						iteration.Id,
						publishedModelName,
						resourceId);

					Console.WriteLine($"Iteration '{iteration.Name}' published.");
					WriteSeparator();
				}
                catch(Exception ex)
                {
					Console.WriteLine($"There was an exception (perhaps nothing changed since last iteration?).");
				}
			}

            if (iteration == null)
            {
                var iterations = await trainingApi.GetIterationsAsync(project.Id);
				iteration = iterations.LastOrDefault();

				Console.WriteLine($"Iteration '{iteration.Name}' found and loaded.");
				WriteSeparator();
			}

			// Prediction Client
			var predictionClient = new CustomVisionPredictionClient()
			{
				ApiKey = customVisionKey,
				Endpoint = endpoint
			};

			// Make predictions against the new project
			Console.WriteLine("Making predictions:");

			var testImages = LoadImagesFromDisk("Test");

			foreach (var image in testImages)
			{
				var imageName = Path.GetFileName((image));

				using (var stream = new MemoryStream(File.ReadAllBytes(image)))
				{
					Console.WriteLine($"\t---------- Image {imageName} ----------");

					var result = await predictionClient.ClassifyImageAsync(
						project.Id,
						publishedModelName,
						stream);

					// Loop over each prediction and write out the results
					foreach (var c in result.Predictions)
					{
						Console.WriteLine($"For Tag {c.TagName}: \t{c.Probability:P3}");
					}
				}
			}

			WriteSeparator();

			Console.WriteLine("Do you want to export the model? (Y/N)");
			var exportModel = Console.ReadKey().Key == ConsoleKey.Y;

			if (exportModel)
			{
				do
				{
					var platform = string.Empty;
					var extension = string.Empty;
					Export export;

					Console.WriteLine("Options: \n1) TensorFlow \n2) CoreML \n3) Other platform \nE) End program");
					var option = Console.ReadKey().Key;

					switch (option)
					{
						case ConsoleKey.D1:
							platform = "TensorFlow";
							extension = "zip";
							break;
						case ConsoleKey.D2:
							platform = "CoreML";
							extension = "mlmodel";
							break;
						case ConsoleKey.D3:
							Console.WriteLine("Type the platform name");
							platform = Console.ReadLine();
							Console.WriteLine($"Now type the file extension for the {platform} exported model.");
							extension = Console.ReadLine();
							break;
						case ConsoleKey.E:
							exportModel = false;
							break;
						default:
							Console.WriteLine("\nOption not supported.");
							break;
					}

					WriteSeparator();

					if (!string.IsNullOrWhiteSpace(platform))
					{
						try
						{
							do
							{
								var exports = await trainingApi.GetExportsAsync(project.Id, iteration.Id);
								export = exports.FirstOrDefault(x => x.Platform == platform);

								if (export == null)
									export = await trainingApi.ExportIterationAsync(project.Id, iteration.Id, platform);

								Thread.Sleep(1000);
								Console.WriteLine($"Status: {export.Status}");
							} while (export.Status == "Exporting");

							Console.WriteLine($"Status: {export.Status}");

							if (export.Status == ExportStatus.Done)
							{
								Console.WriteLine($"Downloading {platform} model");
								var filePath = Path.Combine(Environment.CurrentDirectory, $"{publishedModelName}_{platform}.{extension}");

								using (var httpClient = new HttpClient())
								{
									using (var stream = await httpClient.GetStreamAsync(export.DownloadUri))
									{
										using (var file = new FileStream(filePath, FileMode.Create))
										{
											await stream.CopyToAsync(file);
											Console.WriteLine($"Model exported successfully. Check {filePath}.");
											WriteSeparator();
										}
									}
								}
							}
						}
						catch (Exception ex)
						{
							Console.WriteLine($"Exception found: {ex.Message}");
							WriteSeparator();
						}
					}
				} while (exportModel);
			}

			Console.WriteLine("Press a key to exit the program!");
			Console.ReadKey();
		}

		private static List<string> LoadImagesFromDisk(string directory) => Directory.GetFiles(directory).ToList();

		private static string RepeatCharacter(string s, int n) =>
			new StringBuilder(s.Length * n).AppendJoin(s, new string[n + 1]).ToString();

		private static void WriteSeparator() => Console.WriteLine(RepeatCharacter("-", 30));
	}
}