import sqlite3
import pandas as pd
from faker import Faker

# Initialize Faker
faker = Faker()

# Number of records (scaled up 10x)
num_students = 10000
num_courses = 1000
num_enrollments = 20000  # Adjusted to ensure multiple enrollments per student

# Create SQLite database
conn = sqlite3.connect('synthetic_university.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Students (
        Student_ID INTEGER PRIMARY KEY,
        Name TEXT,
        Age INTEGER,
        Gender TEXT,
        Department TEXT,
        GPA REAL
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Courses (
        Course_ID INTEGER PRIMARY KEY,
        Course_Name TEXT,
        Department TEXT,
        Instructor TEXT,
        Credits INTEGER
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Enrollments (
        Enrollment_ID INTEGER PRIMARY KEY,
        Student_ID INTEGER,
        Course_ID INTEGER,
        Grade TEXT,
        FOREIGN KEY (Student_ID) REFERENCES Students(Student_ID),
        FOREIGN KEY (Course_ID) REFERENCES Courses(Course_ID)
    )
''')

# Generate and insert students data
students = []
student_ids = set()
while len(student_ids) < num_students:
    student_ids.add(faker.random_int(min=1000, max=99999))
for student_id in student_ids:
    students.append((
        student_id,
        faker.name(),
        faker.random_int(min=18, max=25),
        faker.random_element(elements=('Male', 'Female')),
        faker.random_element(elements=('CS', 'Math', 'Physics', 'Chemistry', 'Biology')),
        round(faker.random.uniform(2.0, 4.0), 2)
    ))
cursor.executemany('''
    INSERT INTO Students (Student_ID, Name, Age, Gender, Department, GPA)
    VALUES (?, ?, ?, ?, ?, ?)
''', students)

# Generate and insert courses data
courses = []
course_ids = set()
while len(course_ids) < num_courses:
    course_ids.add(faker.random_int(min=100, max=9999))
for course_id in course_ids:
    courses.append((
        course_id,
        faker.catch_phrase(),
        faker.random_element(elements=('CS', 'Math', 'Physics', 'Chemistry', 'Biology')),
        faker.name(),
        faker.random_int(min=1, max=4)
    ))
cursor.executemany('''
    INSERT INTO Courses (Course_ID, Course_Name, Department, Instructor, Credits)
    VALUES (?, ?, ?, ?, ?)
''', courses)

# Generate and insert enrollments data
enrollments = []
enrollment_ids = set()
while len(enrollment_ids) < num_enrollments:
    enrollment_ids.add(faker.random_int(min=10000, max=999999))
for enrollment_id in enrollment_ids:
    enrollments.append((
        enrollment_id,
        faker.random_element(elements=list(student_ids)),
        faker.random_element(elements=list(course_ids)),
        faker.random_element(elements=('A', 'B', 'C', 'D', 'F'))
    ))
cursor.executemany('''
    INSERT INTO Enrollments (Enrollment_ID, Student_ID, Course_ID, Grade)
    VALUES (?, ?, ?, ?)
''', enrollments)

# Commit changes and close connection
conn.commit()
conn.close()

print("Synthetic university database created successfully.")
