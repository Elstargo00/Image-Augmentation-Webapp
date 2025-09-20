# Image Augmentation App

The Image Augmentation App is a free, open-source tool developed for machine learning enthusiasts to augment their images. This application supports data splitting for training, validation, and testing sets, making it a versatile tool for image dataset preparation.

![webapp_UI](https://github.com/Elstargo00/Image-Augmentation-Webapp/blob/main/ImgAug.png?raw=true)

## Features

- **Data Splitting**: Efficiently divides your dataset into training, validation, and testing sets.
- **Image Augmentation**: Enhances your images with various augmentation techniques.

## Future Enhancements

- **Integration with Labeling Tools**: Plans to integrate with various image labeling tools to streamline the data preparation process.
- **Frontend Development**: Focus on developing a user-friendly frontend and improving the input directory workflow to enhance the user experience.
- **Docker Support**: Aim to containerize the app using Docker for easier distribution and deployment.

## Getting Started
### 1.Clone the repository
Clone the repository to your local machine:

```bash
git clone https://github.com/Elstargo00/Image-Augmentation-Webapp.git
cd Image-Augmentation-Webapp
```

### 2.Create a `.env` file
The `.env` file should contain the necessary environment variables. Example:

```
DJANGO_SECRET_KEY=your-secret-key
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
DJANGO_DEBUG=True
```

Make sure to replace `your-secret-key` with your actual values.

### 3.Build and start the containers
Run the following command to build and start the containers:

```bash
docker-compose up --build
```

### 4.Apply database migrations
Once the containers are running, you need to apple Django migrations to set up
the database:

```bash
docker-compose exec img-aug python manage.py migrate
```

### 5.Access the Django app
Your Django app will now be accessible at `http://localhost:8000`.
