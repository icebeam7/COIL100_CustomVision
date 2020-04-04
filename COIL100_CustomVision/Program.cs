using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction;

namespace COIL100_CustomVision
{
	class Program
	{
		public static async Task Main()
		{
			var labels = new Dictionary<int, string>()
			{
				{ 1, "dristan cold box" },
				{ 2,"onion" },
				{ 4,"tomato" },
				{ 5,"rolaids bottle" },
				{ 7,"arizona iced tea" },
				{ 10,"cup" },
				{ 14,"cat" },
				{ 19,"firetruck car" },
				{ 28,"frog" },
				{ 31,"tylenol" },
				{ 33,"glue" },
				{ 35,"porcelain plate" },
				{ 37,"toy tank" },
				{ 46,"marlboro cigarette box" },
				{ 47,"donut toy" },
				{ 48,"piggy bank" },
				{ 49,"canada dry ginger ale" }
			};

			var projectName = "COIL100 Small";

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

			var trainingApi = new CustomVisionTrainingClient()
			{
				ApiKey = customVisionKey,
				Endpoint = endpoint
			};

			Console.WriteLine("Selecting existing project:");
			var projects = await trainingApi.GetProjectsAsync();
			var project = projects.FirstOrDefault(x => x.Name == projectName);

			var tags = new List<Tag>();

			foreach (var label in labels)
			{
				var tag = await trainingApi.CreateTagAsync(project.Id, label.Value);
				tags.Add(tag);
				Console.WriteLine($"Tag {tag.Name} added");
			}

			Console.WriteLine("\tUploading images");
			var images = LoadImagesFromDisk("Images");

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

			// Now there are images with tags start training the project
			Console.WriteLine("\tTraining");
			var iteration = await trainingApi.TrainProjectAsync(project.Id);

			// The returned iteration will be in progress, and can be queried periodically to see when it has completed
			while (iteration.Status == "Training")
			{
				Thread.Sleep(1000);
				Console.WriteLine($"Iteration status: {iteration.Status}");

				// Re-query the iteration to get it's updated status
				iteration = await trainingApi.GetIterationAsync(project.Id, iteration.Id);
			}

			Console.WriteLine($"Iteration status: {iteration.Status}");

			// The iteration is now trained. Publish it to the prediction endpoint.
			await trainingApi.PublishIterationAsync(project.Id,
				iteration.Id,
				publishedModelName,
				resourceId);
			Console.WriteLine("Done!\n");

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

			Console.WriteLine("Press a key to exit the program!");
			Console.ReadKey();
		}

		private static List<string> LoadImagesFromDisk(string directory) => Directory.GetFiles(directory).ToList();
	}
}