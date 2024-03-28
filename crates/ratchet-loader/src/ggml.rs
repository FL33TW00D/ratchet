use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use half::f16;
use ratchet::{DType, Device, Shape, Tensor};
use std::{
    cell::Cell,
    collections::HashMap,
    io::{BufRead, Seek, SeekFrom},
    mem::MaybeUninit,
};

use crate::{GgmlDType, LoadError};

trait ReadBytesCustom: ReadBytesExt {
    /// Extends to read an exact number of bytes.
    fn read_bytes_with_len(&mut self, len: usize) -> std::io::Result<Vec<u8>> {
        let mut buf: Vec<MaybeUninit<u8>> = Vec::with_capacity(len);
        unsafe {
            buf.set_len(len);
        }
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len) };
        self.read_exact(buf_slice)?;
        let buf = unsafe { std::mem::transmute::<_, Vec<u8>>(buf) };
        Ok(buf)
    }
}
impl<T: std::io::BufRead> ReadBytesCustom for T {}

pub const MAGIC_GGML: u32 = 0x67676d6c;
pub const MAGIC_GGJT: u32 = 0x67676a74;
pub const MAGIC_GGLA: u32 = 0x67676C61;
pub const MAGIC_GGMF: u32 = 0x67676d66;
pub const MAGIC_GGSN: u32 = 0x6767736e;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GGMLFormat {
    GGML(u32),
    GGJT(u32, u32),
    GGLA(u32, u32),
    GGMF(u32, u32),
    GGSN(u32, u32),
}

impl GGMLFormat {
    pub fn read<R: BufRead + Seek>(reader: &mut R) -> Result<GGMLFormat, LoadError> {
        let magic = reader.read_u32::<byteorder::LittleEndian>()?;
        match magic {
            MAGIC_GGML => Ok(GGMLFormat::GGML(magic)),
            _ => {
                let version = reader.read_u32::<byteorder::LittleEndian>()?;
                match magic {
                    MAGIC_GGJT if (1..=3).contains(&version) => {
                        Ok(GGMLFormat::GGJT(magic, version))
                    }
                    MAGIC_GGLA if version == 1 => Ok(GGMLFormat::GGLA(magic, version)),
                    MAGIC_GGMF if version == 1 => Ok(GGMLFormat::GGMF(magic, version)),
                    _ => Err(LoadError::InvalidFormat(magic)),
                }
            }
        }
    }

    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let (magic, version) = match self {
            Self::GGML(magic) => (magic, None),
            Self::GGJT(magic, version) => (magic, Some(version)),
            Self::GGLA(magic, version) => (magic, Some(version)),
            Self::GGMF(magic, version) => (magic, Some(version)),
            Self::GGSN(magic, version) => (magic, Some(version)),
        };

        writer.write_u32::<LittleEndian>(*magic)?;

        if let Some(version) = version {
            writer.write_u32::<LittleEndian>(*version)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TensorHeader {
    pub name: String,
    pub shape: Shape,
    pub dtype: GgmlDType,
    pub start_offset: u64,
    pub numel: usize,
}

impl TensorHeader {
    fn new(name: String, shape: Shape, dtype: GgmlDType, start_offset: u64) -> Self {
        let numel = shape.numel();
        Self {
            name,
            shape,
            dtype,
            start_offset,
            numel,
        }
    }

    pub fn data_size(&self) -> usize {
        self.dtype.tensor_size(self.numel)
    }

    pub fn read_data<R: BufRead + Seek>(&self, reader: &mut R) -> std::io::Result<Vec<u8>> {
        let n_bytes = self.data_size();
        reader.seek(SeekFrom::Start(self.start_offset))?;
        reader.read_bytes_with_len(n_bytes)
    }
}

#[derive(Debug)]
pub struct GGMLModel<M: GGMLCompatible> {
    pub header: M::ModelHeader,
    pub tensors: HashMap<String, TensorHeader>,
    pub total_bytes_loaded: Cell<usize>,
    pub total_loaded: Cell<usize>,
}

impl<M: GGMLCompatible> GGMLModel<M> {
    pub fn new(header: M::ModelHeader, tensors: HashMap<String, TensorHeader>) -> Self {
        Self {
            header,
            tensors,
            total_bytes_loaded: Cell::new(0),
            total_loaded: Cell::new(0),
        }
    }

    pub fn load_tensor<R: BufRead + Seek>(
        &self,
        key: &str,
        reader: &mut R,
        device: &Device,
    ) -> Result<Tensor, LoadError> {
        let header = self.tensors.get(key).ok_or(LoadError::MissingTensor {
            name: key.to_string(),
        })?;
        let mut data = header.read_data(reader)?;
        log::debug!("Loading tensor: {} with size: {} bytes", key, data.len());
        let shape = header.shape.clone();
        let mut dt: DType = header.dtype.into();
        if dt == DType::F16 {
            log::debug!("Casting {} from F16 to F32", key);
            //TODO: terrible cast whilst wgpu doesn't support F16
            let f16_data = bytemuck::cast_slice::<u8, f16>(&data);
            let f32_data = f16_data.iter().map(|f| f.to_f32()).collect::<Vec<_>>();
            data = bytemuck::cast_slice::<f32, u8>(&f32_data).to_vec();
            dt = DType::F32;
        }
        self.total_bytes_loaded
            .set(self.total_bytes_loaded.get() + data.len());
        self.total_loaded.set(self.total_loaded.get() + 1);
        log::debug!(
            "Total bytes loaded: {} bytes",
            self.total_bytes_loaded.get()
        );
        log::debug!("Total tensors loaded: {}", self.total_loaded.get());
        Ok(Tensor::from_bytes(&data, dt, shape, device.clone()).unwrap())
    }
}

struct GGMLLoader {}

impl GGMLLoader {
    pub fn load<R: BufRead + Seek, M: GGMLCompatible>(
        reader: &mut R,
    ) -> Result<GGMLModel<M>, LoadError> {
        let mut tensor_map = HashMap::new();
        let last_position = reader.seek(std::io::SeekFrom::End(0))?;
        reader.seek(std::io::SeekFrom::Start(0))?;
        let model_header = M::load_header(reader)?;

        let mut total_size = 0;
        while reader.stream_position()? != last_position {
            let header = Self::load_single(reader)?;
            total_size += header.data_size() as u64;
            tensor_map.insert(header.name.clone(), header);
        }
        log::debug!("GGML Model size: {}b", total_size);
        Ok(GGMLModel::new(model_header, tensor_map))
    }

    fn load_single<R: BufRead + Seek>(reader: &mut R) -> Result<TensorHeader, LoadError> {
        let n_dims: usize = reader.read_i32::<LittleEndian>()?.try_into()?;
        let name_len = reader.read_i32::<LittleEndian>()?;
        let dtype = reader.read_u32::<LittleEndian>()?;

        let mut dims = vec![0u32; n_dims];
        reader.read_u32_into::<LittleEndian>(&mut dims)?;
        dims.reverse();

        let name = String::from_utf8(reader.read_bytes_with_len(name_len as _)?)?;
        let dtype = GgmlDType::try_from(dtype).map_err(|_| LoadError::UnsupportedDType {
            name: name.clone(),
            dtype,
        })?;

        let start_offset = reader.stream_position()?;
        let header = TensorHeader::new(name, dims.into(), dtype, start_offset);
        let data_size = header.data_size() as u64;
        reader.seek(SeekFrom::Start(start_offset + data_size))?;
        Ok(header)
    }
}

/// # GGML Compatible
///
/// Implement this for your Model if you want to load it from a GGML file.
pub trait GGMLCompatible: Sized {
    type ModelHeader: std::fmt::Debug;

    fn load_header<R: BufRead + Seek>(reader: &mut R) -> Result<Self::ModelHeader, LoadError>;
    fn load_ggml<R: BufRead + Seek>(reader: &mut R) -> Result<GGMLModel<Self>, LoadError> {
        GGMLLoader::load(reader)
    }

    /// Writing is optional
    fn write_header<W: std::io::Write>(_: &Self::ModelHeader, _: &mut W) -> std::io::Result<()> {
        unimplemented!("Writing GGML files is unimplemented for this model")
    }

    fn write_ggml<W: std::io::Write>(
        _model: &GGMLModel<Self>,
        _writer: &mut W,
    ) -> std::io::Result<()> {
        todo!()
    }

    fn write_tensor<W: std::io::Write>(
        name: &str,
        tensor: Tensor,
        writer: &mut W,
    ) -> std::io::Result<usize> {
        let mut shape = tensor.shape().clone();
        let dtype = tensor.dt().to_u32();
        let n_dims = shape.rank();
        writer.write_i32::<LittleEndian>(n_dims as i32)?;
        writer.write_i32::<LittleEndian>(name.len() as i32)?;
        writer.write_u32::<LittleEndian>(dtype)?;

        shape.reverse();
        for dim in shape.iter() {
            writer.write_u32::<LittleEndian>(*dim as _)?;
        }
        writer.write_all(name.as_bytes())?;
        let data = unsafe {
            tensor
                .into_bytes()
                .map_err(|_| std::io::ErrorKind::InvalidData)?
        };
        log::info!("Writing tensor: {} with size {} bytes", name, data.len());
        writer.write_all(&data)?;
        Ok(data.len())
    }
}
