import './App.css';
import Button from 'react-bootstrap/Button';
import Container from 'react-bootstrap/Container';
import Navbar from 'react-bootstrap/Navbar';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Card from 'react-bootstrap/Card';
import Modal from 'react-bootstrap/Modal';
import CanvasDraw from "react-canvas-draw";
import { useEffect, useRef, useState } from 'react';

const style = {
  preview_search_image: {
    height: 300,
    objectFit: "contain"
  }
}


function MyVerticallyCenteredModal(props) {
  return (
    <Modal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
          Building Plan {props.idx}
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <img src={props.url} style={{margin: "0 auto", width: 500, height:500, objectFit: 'contain'}}/>
        <h4>Contact Information</h4>
        <p>
          Thank you for your interest in this building plan. Please contact the owner for further details.
        </p>
        <p>
          Price: Upon Request
        </p>
        <p>
          Phone: +(343) 25487/855469
        </p>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={props.onHide}>Contact the Owner</Button>
      </Modal.Footer>
    </Modal>
  );
}


function SearchResults({ searchResults, onClickHandler=null}) {
  const [imgUrls, setImageUrls] = useState(null);
  
  useEffect(() => {
    if (searchResults) {
      Promise.all(
        searchResults.map(file =>
          fetch(`http://localhost:5000/api/image?image=${file}`)
            .then(res => res.blob())
            .then(b => URL.createObjectURL(b))
        )
      ).then(urls => setImageUrls(urls))
    }
  
  }, [ searchResults ])

  return (
    <>
      {
        (imgUrls)
          ? imgUrls.map((url, idx)=> 
            <Card key={idx}  style={{ width: '18rem', marginRight: 8, marginTop: 8}} onClick={() => onClickHandler ? onClickHandler(idx, url): ""}>
              <Card.Body>
                <img src={url} style={{width: 200, height: 200, objectFit: "contain"}}></img>
              </Card.Body>
            </Card>)
          : ""
      }
    </>
  )

}


function App() {
  const [imgUrl, setImgUrl] = useState("https://via.placeholder.com/200x200");
  const [searchResults, setSearchResults] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [currentImage, setCurrentImage] = useState("");
  const [currentIdx, setCurrentIdx] = useState(0);
  const [modalShow, setModalShow] = useState(false);

  const canvasRef = useRef(null);

  function displayImage(event) {
    setImgUrl(URL.createObjectURL(event.target.files[0]));
    setSelectedFile(event.target.files[0]);
  }

  function fetchSearchResults(event) {
    if (selectedFile) {
      const formData = new FormData()
  
      formData.append('image', selectedFile)

      fetch(`http://localhost:5000/api/search`, {
        method: 'POST',
        body: formData
      })
      .then(res => res.json())
      .then(data => setSearchResults(data.content))
      .catch(error => {
        console.warn(error);
      });
    }
  }

  function handleDrawing(event, canvasRef) {
    if (canvasRef.current) {
      const dataUrl = canvasRef.current.canvasContainer.children[1].toDataURL();

      fetch('http://localhost:5000/api/drawing', {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          img: dataUrl
        })
      })
      .then(res => res.json())
      .then(data => setSearchResults(data.content))
      .catch(error => {
        console.warn(error);
      });
      
    }
  }

  return (
    <>
      <Navbar bg="light">
        <Container >
          <Navbar.Brand href="#home">GeoPho</Navbar.Brand>
        </Container>
      </Navbar>
      <Container fluid className="masthead">
        <Row className="justify-content-center mt-2 mb-4">
            <Col className="text-center">
              <h1>Where great ideas take shape</h1>
            </Col>
          </Row>
          <Row>
            <Col>
              <CanvasDraw ref={canvasRef} canvasHeight={450} canvasWidth={450} brushColor="#000000FF" brushRadius={3} gridColor="#FFFFFFFF"/>
              <Button className="btn-primary btn-lg mt-4" onClick={evt => handleDrawing(evt, canvasRef)}>Find Similar Form</Button>
            </Col>
            <Col className="text-center">
              <h1>or</h1>
            </Col>
            <Col className="text-center">
              <Card style={{width: 600, height: 450, background: "#FFFFFF50"}}>
                <img className="card-img-top mt-4" src={imgUrl} style={style.preview_search_image}/>
                <Card.Body>
                  <label for="file-upload" 
                      class="custom-file-upload btn-outline-secondary mt-2">
                      <i class="fa fa-cloud-upload"></i> Upload a Floorplan
                  </label>
                  <input
                      id="file-upload"
                      onChange={event => displayImage(event)}
                      type="file"
                    />
                  <br/>
                </Card.Body>
              </Card>

              <button className="btn btn-primary card-text mt-4 btn-lg" onClick={fetchSearchResults}>Search</button>
            </Col>
            <MyVerticallyCenteredModal
              show={modalShow}
              url={currentImage}
              idx={currentIdx}
              onHide={() => setModalShow(false)}
            />
          </Row>
      </Container>
      <Container>
        
        <Row style={{marginTop: 24}}>
          { (searchResults) 
              ? <SearchResults searchResults={searchResults} onClickHandler={(idx, url) => {
                setModalShow(true);
                setCurrentImage(url);
                setCurrentIdx(idx);
              }}/>
              : "Nothing to display"
          }
        </Row>
      </Container>
    </>
  );
}

export default App;
