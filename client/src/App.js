import React from 'react';
import { useState } from 'react';
import {
  Box,
  Text,
  ChakraProvider,
  VStack,
  Input,
  theme,
  Button,
  InputLeftAddon,
  InputGroup,
  Select,
  Spinner,
} from '@chakra-ui/react';
import bg from './assets/bg.webp';

function App() {
  const [caratValue, setCaratValue] = React.useState('');
  const [depthValue, setDepthValue] = React.useState('');
  const [tableValue, setTableValue] = React.useState('');
  const [xValue, setXValue] = React.useState('');
  const [yValue, setYValue] = React.useState('');
  const [zValue, setZValue] = React.useState('');
  const [cutValue, setCutValue] = React.useState('');
  const [colorValue, setColorValue] = React.useState('');
  const [clarityValue, setClarityValue] = React.useState('');
  const [price, setPrice] = React.useState('$');
  const handleChanges = (e, setVal) => setVal(e.target.value);
  const [loading, setLoading] = useState(false);
  const onSubmit = () => {
    if (
      caratValue === '' ||
      depthValue === '' ||
      tableValue === '' ||
      xValue === '' ||
      yValue === '' ||
      zValue === '' ||
      cutValue === '' ||
      colorValue === '' ||
      clarityValue === ''
    ) {
      alert('Please fill all the fields');
      return;
    } else {
      setLoading(true);
      fetch('https://diamondpricepredictorapp.azurewebsites.net/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          carat: caratValue,
          depth: depthValue,
          table: tableValue,
          x: xValue,
          y: yValue,
          z: zValue,
          cut: cutValue,
          color: colorValue,
          clarity: clarityValue,
        }),
      })
        .then(res => res.json())
        .then(data => {
          setPrice(data.price);
          setCaratValue('');
          setDepthValue('');
          setTableValue('');
          setXValue('');
          setYValue('');
          setZValue('');
          setCutValue('');
          setColorValue('');
          setClarityValue('');
          setLoading(false);
          console.log(data);
        })
        .catch(err => console.log(err));
    }
  };

  return (
    <ChakraProvider theme={theme}>
      <Box
        display={'flex'}
        alignItems={'center'}
        w={'full'}
        minH={'100vh'}
        h={'fit-content'}
        backgroundImage={bg}
        opacity={loading ? '0.5' : '1'}
        backgroundRepeat={'no-repeat'}
        backgroundSize={'cover'}
        p={['10px', '10px', '8vh', '10vh']}
      >
        {loading ? (
          <Spinner  ml={'42vw'} color="white" size="xl" />
        ) : (
          <VStack
            mx={'auto'}
            display={'flex'}
            alignItems={'center'}
            p={['20px', '20px', '80px', '80px']}
            px={['20px', '20px', '100px', '100px']}
            w={['full', 'full', 'fit-content', 'fit-content']}
            h={['full', 'full', 'fit-content', 'fit-content']}
            borderRadius={['10px', '10px', '30px', '30px']}
            size={'container.xl'}
            opacity={['0.7', '0.7', '0.5', '0.5']}
            bgColor={'#cbcfd4'}
          >
            <InputGroup>
              <InputLeftAddon
                border={'none'}
                bgColor={'#cbcfd4'}
                fontSize={'lg'}
                fontWeight={'bold'}
                color={'black'}
                children="C : "
              />
              <Input
                _placeholder={{ opacity: 1, color: 'black' }}
                variant={'filled'}
                border={'1px solid #cbcfd4'}
                borderBottom={'1px solid black'}
                w={['51vw', '40vw', '30vw']}
                bgColor={'#cbcfd4'}
                mb={'8px'}
                value={caratValue}
                onChange={e => handleChanges(e, setCaratValue)}
                focusBorderColor="black"
                minW={'fit-content'}
                textAlign={'center'}
                placeholder="Enter carat value (float)"
                color={'black'}
              />
            </InputGroup>
            <InputGroup>
              <InputLeftAddon
                border={'none'}
                bgColor={'#cbcfd4'}
                fontSize={'lg'}
                color={'black'}
                fontWeight={'bold'}
                children="D : "
              />
              <Input
                _placeholder={{ opacity: 1, color: 'black' }}
                variant={'filled'}
                bgColor={'#cbcfd4'}
                mb={'8px'}
                value={depthValue}
                onChange={e => handleChanges(e, setDepthValue)}
                focusBorderColor="black"
                border={'1px solid #cbcfd4'}
                borderBottom={'1px solid black'}
                w={['51vw', '40vw', '30vw']}
                minW={'fit-content'}
                textAlign={'center'}
                placeholder="Enter depth value (float)"
                color={'black'}
              />
            </InputGroup>
            <InputGroup>
              <InputLeftAddon
                border={'none'}
                bgColor={'#cbcfd4'}
                color={'black'}
                fontSize={'lg'}
                fontWeight={'bold'}
                children="T : "
              />
              <Input
                _placeholder={{ opacity: 1, color: 'black' }}
                variant={'filled'}
                bgColor={'#cbcfd4'}
                mb={'8px'}
                value={tableValue}
                onChange={e => handleChanges(e, setTableValue)}
                focusBorderColor="black"
                border={'1px solid #cbcfd4'}
                borderBottom={'1px solid black'}
                w={['51vw', '40vw', '30vw']}
                minW={'fit-content'}
                textAlign={'center'}
                placeholder="Enter table value (float)"
                color={'black'}
              />
            </InputGroup>
            <InputGroup>
              <InputLeftAddon
                border={'none'}
                color={'black'}
                bgColor={'#cbcfd4'}
                fontSize={'lg'}
                fontWeight={'bold'}
                children="X : "
              />
              <Input
                _placeholder={{ opacity: 1, color: 'black' }}
                variant={'filled'}
                bgColor={'#cbcfd4'}
                value={xValue}
                onChange={e => handleChanges(e, setXValue)}
                mb={'8px'}
                focusBorderColor="black"
                border={'1px solid #cbcfd4'}
                borderBottom={'1px solid black'}
                w={['51vw', '40vw', '30vw']}
                minW={'fit-content'}
                textAlign={'center'}
                placeholder="Enter X value (float)"
                color={'black'}
              />
            </InputGroup>
            <InputGroup>
              <InputLeftAddon
                border={'none'}
                bgColor={'#cbcfd4'}
                color={'black'}
                fontSize={'lg'}
                fontWeight={'bold'}
                children="Y : "
              />
              <Input
                _placeholder={{ opacity: 1, color: 'black' }}
                variant={'filled'}
                bgColor={'#cbcfd4'}
                mb={'8px'}
                value={yValue}
                onChange={e => handleChanges(e, setYValue)}
                focusBorderColor="black"
                border={'1px solid #cbcfd4'}
                borderBottom={'1px solid black'}
                w={['51vw', '40vw', '30vw']}
                minW={'fit-content'}
                textAlign={'center'}
                placeholder="Enter Y value (float)"
                color={'black'}
              />
            </InputGroup>
            <InputGroup>
              <InputLeftAddon
                border={'none'}
                bgColor={'#cbcfd4'}
                color={'black'}
                fontSize={'lg'}
                fontWeight={'bold'}
                children="Z : "
              />
              <Input
                _placeholder={{ opacity: 1, color: 'black' }}
                variant={'filled'}
                bgColor={'#cbcfd4'}
                mb={'8px'}
                value={zValue}
                onChange={e => handleChanges(e, setZValue)}
                focusBorderColor="black"
                border={'1px solid #cbcfd4'}
                borderBottom={'1px solid black'}
                w={['51vw', '40vw', '30vw']}
                minW={'fit-content'}
                textAlign={'center'}
                placeholder="Enter Z value (float)"
                color={'black'}
              />
            </InputGroup>
            <Select
              mt={'10px'}
              w={['50vw', '40vw', '20vw']}
              focusBorderColor="#cbcfd4"
              variant={'filled'}
              color={'black'}
              bgColor={['white']}
              value={cutValue}
              onChange={e => handleChanges(e, setCutValue)}
              placeholder="Select Cut"
            >
              <option value="Fair">Fair</option>
              <option value="Good">Good</option>
              <option value="Very Good">Very Good</option>
              <option value="Premium">Premium</option>
              <option value="Ideal">Ideal</option>
            </Select>
            <Select
              w={['50vw', '40vw', '20vw']}
              focusBorderColor="#cbcfd4"
              variant={'filled'}
              value={colorValue}
              color={'black'}
              bgColor={['white']}
              onChange={e => handleChanges(e, setColorValue)}
              placeholder="Select Color"
            >
              <option value="D">D</option>
              <option value="E">E</option>
              <option value="F">F</option>
              <option value="G">G</option>
              <option value="H">H</option>
              <option value="I">I</option>
              <option value="J">J</option>
            </Select>

            <Select
              w={['50vw', '40vw', '20vw']}
              focusBorderColor="#cbcfd4"
              variant={'filled'}
              color={'black'}
              bgColor={['white']}
              placeholder="Select Clarity"
              value={clarityValue}
              onChange={e => handleChanges(e, setClarityValue)}
            >
              <option value="I1">I1</option>
              <option value="SI2">SI2</option>
              <option value="SI1">SI1</option>
              <option value="VS2">VS2</option>
              <option value="VS1">VS1</option>
              <option value="VVS2">VVS2</option>
              <option value="VVS1">VVS1</option>
              <option value="IF">IF</option>
            </Select>

            <Button
              mt={'10px'}
              _hover={{ opacity: '0.8' }}
              bgColor={'gray.900'}
              color={'white'}
              type="submit"
              onClick={onSubmit}
              variant={'ghost'}
            >
              Submit
            </Button>

            <Text
              mt={'20px'}
              fontWeight={'bold'}
              color={'blackAlpha.900'}
              fontSize={'2xl'}
            >
              Price : ${price}
            </Text>
          </VStack>
        )}
      </Box>
    </ChakraProvider>
  );
}

export default App;
