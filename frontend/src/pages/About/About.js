import React from 'react'

import './About.css'

export default function info() {
    return (
        <div className="info-container">
            <h1 className="info-heading">info</h1>
            <div className="info-main">
                <p className="info-content">
                    This is an realtime AI based Yoga Trainer which detects your pose how well you are doing.
                    
                    This AI first predicts keypoints or coordinates of different parts of the body(basically where
                    they are present in an image) and then it use another classification model to classify the poses if 
                    someone is doing a pose and if AI detects that pose more than 95% probability and then it will notify you are 
                    doing correctly(by making virtual skeleton green). I have used Tensorflow pretrained Movenet Model To Predict the 
                    Keypoints and building a neural network top of that which uses these coordinates and classify a yoga pose.

                    I have trained the model in python because of tensorflowJS we can leverage the support of browser so I converted 
                    the keras/tensorflow model to tensorflowJS.
                </p>
                <div className="developer-info">
                    <h4>info Developer</h4>
                    <p className="about-content">I am kk, I am Full Stack Developer, AI Enthusiastic, Content Creator, Tutor,
                    </p>
                    <h4>Contact</h4>
                    <a href="https://www.instagram.com/codedharsh75/"><p className="about-content">Instagram</p></a>
                    <a href="https://www.youtube.com/channel/UCiD7kslR7lKSaPGSQ-heOWg"><p  className="about-content">Youtube</p></a>
                    <a href="https://github.com/harshbhatt7585"><p  className="about-content">GitHub</p></a>
                </div>
            </div>
        </div>
    )
}
