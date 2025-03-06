import React from "react";
import daniel from './img/Daniel.jpeg'
import etinosa from './img/Etinosa.jpeg'
import okosun from './img/Supervisor.jpg'

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white text-center p-4 mt-8">
      <div className="container mx-auto">
        <div className="mb-5">Project by</div>
        <div className="flex flex-row justify-between items-center">
            <div className="flex flex-col justify-between items-center">
                <img className="rounded-full lg:w-32 lg:h-32 w-20 h-20" src={daniel}/>
                <p className="text-sm">Daniel Chukwuemeke Eni (Project Student)</p>
            </div>
            <div className="flex flex-col justify-between items-center">
                <img className="rounded-full lg:w-32 lg:h-32 w-20 h-20" src={etinosa}/>
                <p className="text-sm">Etinosa Prosper Emmanuel  (Project Student)</p>
            </div>
            <div className="flex flex-col justify-between items-center">
                <img className="rounded-full lg:w-32 lg:h-32 w-20 h-20" src={okosun}/>
                <p className="text-sm">Dr Okosun Oduware(Project Supervisor)</p>
            </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
